use std::{
    env,
    fs::{self},
    path::PathBuf,
};

use heck::{ToPascalCase, ToSnakeCase};
use proc_macro2::TokenStream;
use shader_slang::Downcast;

fn main() {
    // println!("cargo:rerun-if-changed=shaders");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    let global_session = shader_slang::GlobalSession::new().unwrap();
    let search_path = std::ffi::CString::new("shaders").unwrap();

    let opts = shader_slang::CompilerOptions::default()
        .optimization(shader_slang::OptimizationLevel::Maximal)
        .matrix_layout_column(true)
        .glsl_force_scalar_layout(true);
    let targets = [shader_slang::TargetDesc::default()
        .format(shader_slang::CompileTarget::Spirv)
        .profile(global_session.find_profile("spirv_1_5"))];
    let search_paths = [search_path.as_ptr()];

    let session_desc = shader_slang::SessionDesc::default()
        .targets(&targets)
        .search_paths(&search_paths)
        .options(&opts);
    let session = global_session.create_session(&session_desc).unwrap();

    let mut files = Vec::new();
    for entry in walkdir::WalkDir::new("shaders") {
        let path = entry.unwrap().path().to_path_buf();
        if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("slang") {
            files.push(path.to_path_buf());
        }
    }
    files.sort();

    struct ModuleInfo {
        def: TokenStream,
        field_ident: proc_macro2::Ident,
        type_ident: proc_macro2::Ident,
    }
    let mut modules = Vec::new();

    fn field_ident(name: &str) -> proc_macro2::Ident {
        let snake = name.to_ascii_lowercase().to_snake_case();
        quote::format_ident!("{}", &snake)
    }
    fn type_ident(name: &str) -> proc_macro2::Ident {
        let pascal = name.to_ascii_lowercase().to_pascal_case();
        quote::format_ident!("{}", &pascal)
    }

    for file in files {
        let module_name = file
            .strip_prefix("shaders")
            .unwrap()
            .with_extension("")
            .to_string_lossy()
            .replace('\\', "/");
        let module = session.load_module(&module_name).unwrap();

        struct EntryInfo {
            ident: proc_macro2::Ident,
            entry_name: String,
            spv_name: String,
        }
        let mut entries = Vec::new();

        for entry_point in module.entry_points() {
            let entry_name = entry_point.function_reflection().name().to_owned();
            let program = session
                .create_composite_component_type(&[
                    module.downcast().clone(),
                    entry_point.downcast().clone(),
                ])
                .unwrap();
            let linked = program.link().unwrap();
            let reflection = linked.layout(0).unwrap();

            let spv_base = format!("{}_{}", module_name, entry_name).to_ascii_lowercase();
            let spv_name = format!("{}.spv", spv_base);
            let out_path = out_dir.join(&spv_name);

            let bytecode = linked.entry_point_code(0, 0).unwrap();
            fs::write(&out_path, bytecode.as_slice()).unwrap();

            entries.push(EntryInfo {
                ident: field_ident(&entry_name),
                entry_name,
                spv_name,
            })
        }

        if entries.is_empty() {
            continue;
        }
        let module_field_ident = field_ident(&module_name);
        let module_type_ident = type_ident(&module_name);

        let entry_field_defs = entries.iter().map(|e| &e.ident);
        let entry_field_inits = entries.iter().map(|e| &e.ident);

        let spv_names = entries
            .iter()
            .map(|e| proc_macro2::Literal::string(&e.spv_name));

        let def = quote::quote! {
            #[derive(Debug)]
            pub struct #module_type_ident {
                #(pub #entry_field_defs: vk::ShaderModule,)*
            }
            impl CreateModule for #module_type_ident {
                fn new(device: &ash::Device) -> Self {
                    Self {
                        #(
                            #entry_field_inits: {
                                let bytes = std::include_bytes!(concat!(
                                    env!("OUT_DIR"),
                                    "/",
                                    #spv_names
                                ));
                                let src: Vec<u32> = bytes
                                    .chunks_exact(4)
                                    .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                                    .collect();
                                unsafe {
                                    device
                                        .create_shader_module(
                                            &vk::ShaderModuleCreateInfo::default().code(&src),
                                            None,
                                        )
                                        .unwrap()
                                }
                            },
                        )*
                    }
                }
            }
        };
        modules.push(ModuleInfo {
            def,
            type_ident: module_type_ident,
            field_ident: module_field_ident,
        })
    }

    let module_defs = modules.iter().map(|m| &m.def);
    let module_field_defs = modules.iter().map(|m| &m.field_ident);
    let module_field_inits = modules.iter().map(|m| &m.field_ident);
    let module_type_defs = modules.iter().map(|m| &m.type_ident);
    let module_type_inits = modules.iter().map(|m| &m.type_ident);

    let code = quote::quote! {
        pub mod shaders {
            use ash::vk::{self};

            #(#module_defs)*

            pub trait CreateModule: Sized {
                fn new(device: &ash::Device) -> Self;
            }

            #[derive(Debug)]
            pub struct Shaders {
                #(
                    pub #module_field_defs: #module_type_defs,
                )*
            }

            impl Shaders {
                pub fn new(device: &ash::Device) -> Self {
                    Self {
                        #(
                            #module_field_inits: #module_type_inits::new(device),
                        )*
                    }
                }
            }
        }
    };

    let ast = syn::parse2(code).unwrap();
    let formatted = prettyplease::unparse(&ast);

    let out_path = out_dir.join("shaders.rs");
    fs::write(&out_path, formatted).unwrap();
}
