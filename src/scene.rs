use ash::vk::{self};
use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Mat4, Quat, Vec3};
use vk_mem::Alloc;

use crate::utils::{
    buffer::Buffer,
    context::VkCtx,
    image::{Image, image},
};

pub struct SceneResources {
    pub vertex_buffers: Vec<Buffer>,
    pub index_buffers: Vec<Buffer>,
    pub index_counts: Vec<u32>,
    pub images: Vec<Image>,
    pub samplers: Vec<vk::Sampler>,
    pub object_buffer: Buffer,
    pub material_buffer: Buffer,
    pub primitives: Vec<PrimitiveData>,
    pub vertices_count: u32,
    pub triangles_count: u32,
}

#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
#[repr(C)]
pub struct ObjectData {
    pub transform: [[f32; 4]; 4],
    pub normal_transform: [[f32; 3]; 3],
}

#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
#[repr(C)]
struct MaterialData {
    pub albedo_factor: [f32; 4],
    pub emissive_factor: [f32; 3],
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub alpha_cutoff: f32,
    pub base_color_texture: i32,
    pub metallic_roughness_texture: i32,
    pub normal_texture: i32,
    pub occlusion_texture: i32,
    pub emissive_texture: i32,
    pub albedo_sampler_index: u32,
    pub metallic_roughness_sampler_index: u32,
    pub normal_sampler_index: u32,
    pub occlusion_sampler_index: u32,
    pub emissive_sampler_index: u32,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct PrimitiveData {
    pub object_id: u32,
    pub material_id: u32,
    pub bounds: [glam::Vec3; 8],
}

#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct VertexData {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub tangent: [f32; 4],
    pub color: [f32; 3],
    pub uv: [f32; 2],
}

const ANISOTROPIC_SAMPLES: f32 = 16.0;

impl SceneResources {
    pub fn create(ctx: &VkCtx) -> Self {
        let (document, buffers, textures) = gltf::import("assets/sponza.glb").unwrap();
        let scene = document.default_scene().unwrap();

        fn walk_transform(node: gltf::Node, parent: Mat4, out: &mut [Mat4]) {
            let local = Mat4::from_cols_array_2d(&node.transform().matrix());
            let transform = parent * local;
            out[node.index()] = transform;
            for child in node.children() {
                walk_transform(child, transform, out);
            }
        }
        let mut transforms = vec![Mat4::IDENTITY; document.nodes().len()];
        for root in scene.nodes() {
            let base_transform = Mat4::from_scale_rotation_translation(
                Vec3::splat(1.0),
                Quat::from_axis_angle(Vec3::X, 90.0f32.to_radians()),
                Vec3::ZERO,
            );
            walk_transform(root, base_transform, &mut transforms);
        }

        let object_buffer = {
            let data = transforms
                .iter()
                .map(|transform| ObjectData {
                    transform: transform.to_cols_array_2d(),
                    normal_transform: Mat3::from_mat4(transform.clone())
                        .inverse()
                        .transpose()
                        .to_cols_array_2d(),
                })
                .collect::<Vec<_>>();
            Buffer::from_data(
                ctx,
                vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::STORAGE_BUFFER,
                bytemuck::cast_slice(&data),
            )
        };

        let mut samplers = Vec::new();
        const ADDRESS_MODES: [vk::SamplerAddressMode; 3] = [
            vk::SamplerAddressMode::CLAMP_TO_EDGE,
            vk::SamplerAddressMode::REPEAT,
            vk::SamplerAddressMode::MIRRORED_REPEAT,
        ];
        for address_mode_u in ADDRESS_MODES {
            for address_mode_v in ADDRESS_MODES {
                samplers.push(unsafe {
                    ctx.device
                        .create_sampler(
                            &vk::SamplerCreateInfo::default()
                                .address_mode_u(address_mode_u)
                                .address_mode_v(address_mode_v)
                                .mag_filter(vk::Filter::LINEAR)
                                .min_filter(vk::Filter::LINEAR)
                                .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                                .min_lod(0.0)
                                .max_lod(vk::LOD_CLAMP_NONE)
                                .anisotropy_enable(true)
                                .max_anisotropy(ANISOTROPIC_SAMPLES.min(
                                    ctx.physical_device_properties.limits.max_sampler_anisotropy,
                                )),
                            None,
                        )
                        .unwrap()
                });
            }
        }
        fn sampler_index(sampler: &gltf::texture::Sampler<'_>) -> u32 {
            const fn wrap_index(mode: gltf::texture::WrappingMode) -> u32 {
                match mode {
                    gltf::texture::WrappingMode::ClampToEdge => 0,
                    gltf::texture::WrappingMode::Repeat => 1,
                    gltf::texture::WrappingMode::MirroredRepeat => 2,
                }
            }
            wrap_index(sampler.wrap_s()) * 3 + wrap_index(sampler.wrap_t())
        }

        let material_buffer = {
            let data = document
                .materials()
                .into_iter()
                .map(|mat| {
                    let pbr = mat.pbr_metallic_roughness();
                    MaterialData {
                        albedo_factor: pbr.base_color_factor(),
                        emissive_factor: mat.emissive_factor(),
                        metallic_factor: pbr.metallic_factor(),
                        roughness_factor: pbr.roughness_factor(),
                        alpha_cutoff: mat.alpha_cutoff().unwrap_or(0.5),
                        base_color_texture: pbr
                            .base_color_texture()
                            .map(|t| t.texture().source().index() as i32)
                            .unwrap_or(-1),
                        metallic_roughness_texture: pbr
                            .metallic_roughness_texture()
                            .map(|t| t.texture().source().index() as i32)
                            .unwrap_or(-1),
                        normal_texture: mat
                            .normal_texture()
                            .map(|t| t.texture().source().index() as i32)
                            .unwrap_or(-1),
                        occlusion_texture: mat
                            .occlusion_texture()
                            .map(|t| t.texture().source().index() as i32)
                            .unwrap_or(-1),
                        emissive_texture: mat
                            .emissive_texture()
                            .map(|t| t.texture().source().index() as i32)
                            .unwrap_or(-1),
                        albedo_sampler_index: pbr
                            .base_color_texture()
                            .map(|t| sampler_index(&t.texture().sampler()))
                            .unwrap_or(0),
                        metallic_roughness_sampler_index: pbr
                            .metallic_roughness_texture()
                            .map(|t| sampler_index(&t.texture().sampler()))
                            .unwrap_or(0),
                        normal_sampler_index: mat
                            .normal_texture()
                            .map(|t| sampler_index(&t.texture().sampler()))
                            .unwrap_or(0),
                        occlusion_sampler_index: mat
                            .occlusion_texture()
                            .map(|t| sampler_index(&t.texture().sampler()))
                            .unwrap_or(0),
                        emissive_sampler_index: mat
                            .emissive_texture()
                            .map(|t| sampler_index(&t.texture().sampler()))
                            .unwrap_or(0),
                    }
                })
                .collect::<Vec<_>>();
            Buffer::from_data(
                ctx,
                vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::STORAGE_BUFFER,
                bytemuck::cast_slice(&data),
            )
        };

        let mut images_srgb = vec![false; textures.len()];
        for mat in document.materials() {
            if let Some(tex) = mat.pbr_metallic_roughness().base_color_texture() {
                images_srgb[tex.texture().source().index()] = true;
            }
            if let Some(tex) = mat.emissive_texture() {
                images_srgb[tex.texture().source().index()] = true;
            }
        }

        let mut images = Vec::new();
        for (i, img) in textures.iter().enumerate() {
            let format = if images_srgb[i] {
                vk::Format::R8G8B8A8_SRGB
            } else {
                vk::Format::R8G8B8A8_UNORM
            };
            let pixels = match img.format {
                gltf::image::Format::R8G8B8A8 => img.pixels.clone(),
                gltf::image::Format::R8G8B8 => {
                    let mut pixels = Vec::with_capacity(img.pixels.len() * 4 / 3);
                    for rgb in img.pixels.as_chunks::<3>().0 {
                        pixels.extend_from_slice(&[rgb[0], rgb[1], rgb[2], 255]);
                    }
                    pixels
                }
                gltf::image::Format::R8G8 => {
                    let mut pixels = Vec::with_capacity(img.pixels.len() * 2);
                    for rg in img.pixels.as_chunks::<2>().0 {
                        pixels.extend_from_slice(&[rg[0], rg[1], 0, 255]);
                    }
                    pixels
                }
                gltf::image::Format::R8 => {
                    let mut pixels = Vec::with_capacity(img.pixels.len() * 4);
                    for r in img.pixels.iter() {
                        pixels.extend_from_slice(&[*r, 0, 0, 255]);
                    }
                    pixels
                }
                f => panic!("bad format {:#?}", f),
            };
            let mip_levels = img.width.max(img.height).ilog2() + 1;

            let mut image = image()
                .extent_2d(vk::Extent2D {
                    width: img.width,
                    height: img.height,
                })
                .format(format)
                .mip_levels(mip_levels)
                .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_SRC)
                .create_with_data(
                    ctx,
                    vk_mem::AllocationCreateFlags::DEDICATED_MEMORY,
                    &pixels,
                );

            image.generate_mipmaps(ctx, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

            images.push(image);
        }

        let mut vertex_buffers = Vec::new();
        let mut index_buffers = Vec::new();
        let mut index_counts = Vec::new();
        let mut primitives = Vec::new();
        let mut vertices_count = 0;
        let mut triangles_count = 0;

        for node in document.nodes() {
            let Some(mesh) = node.mesh() else {
                continue;
            };
            for primitive in mesh.primitives() {
                let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

                let positions = reader.read_positions().unwrap();

                let vertex_count = positions.len();
                let normals = reader.read_normals().unwrap();
                // let Some(tangents) = reader.read_tangents() else {
                //     eprintln!("no tangents for primitive {}, skipping", primitive.index());
                //     continue;
                // };
                let tangents: Box<dyn ExactSizeIterator<Item = [f32; 4]> + '_> = match reader
                    .read_tangents()
                {
                    Some(tangents) => Box::new(tangents),
                    None => Box::new(std::iter::repeat([1.0f32, 0.0, 0.0, 1.0]).take(vertex_count)),
                };
                let colors: Box<dyn ExactSizeIterator<Item = [f32; 3]> + '_> =
                    match reader.read_colors(0) {
                        Some(colors) => Box::new(colors.into_rgb_f32()),
                        None => Box::new(std::iter::repeat([1.0f32, 1.0, 1.0]).take(vertex_count)),
                    };
                let tex_coords: Box<dyn ExactSizeIterator<Item = [f32; 2]> + '_> =
                    match reader.read_tex_coords(0) {
                        Some(uvs) => Box::new(uvs.into_f32()),
                        None => Box::new(std::iter::repeat([0.0, 0.0]).take(vertex_count)),
                    };

                let vertices = positions
                    .into_iter()
                    .zip(normals)
                    .zip(tangents)
                    .zip(colors)
                    .zip(tex_coords)
                    .map(|((((position, normal), tangent), color), uv)| VertexData {
                        position,
                        normal,
                        tangent,
                        color,
                        uv,
                    })
                    .collect::<Vec<_>>();

                let indices = reader
                    .read_indices()
                    .unwrap()
                    .into_u32()
                    .collect::<Vec<_>>();

                vertices_count += vertices.len() as u32;
                triangles_count += indices.len() as u32 / 3;

                let transform = &transforms[node.index()];
                let mut aabb_min = glam::Vec3::MAX;
                let mut aabb_max = glam::Vec3::MIN;
                for vertex in &vertices {
                    let pos = glam::Vec3::from_array(vertex.position);
                    aabb_min = aabb_min.min(pos);
                    aabb_max = aabb_max.max(pos);
                }
                let bounds = [
                    glam::vec3(aabb_min.x, aabb_min.y, aabb_min.z),
                    glam::vec3(aabb_min.x, aabb_min.y, aabb_max.z),
                    glam::vec3(aabb_min.x, aabb_max.y, aabb_min.z),
                    glam::vec3(aabb_min.x, aabb_max.y, aabb_max.z),
                    glam::vec3(aabb_max.x, aabb_min.y, aabb_min.z),
                    glam::vec3(aabb_max.x, aabb_min.y, aabb_max.z),
                    glam::vec3(aabb_max.x, aabb_max.y, aabb_min.z),
                    glam::vec3(aabb_max.x, aabb_max.y, aabb_max.z),
                ]
                .map(|corner| transform.transform_point3(corner));

                primitives.push(PrimitiveData {
                    object_id: node.index() as u32,
                    material_id: primitive.material().index().unwrap_or(0) as u32,
                    bounds,
                });

                vertex_buffers.push(Buffer::from_data(
                    ctx,
                    vk::BufferUsageFlags::VERTEX_BUFFER,
                    bytemuck::cast_slice(&vertices),
                ));

                index_buffers.push(Buffer::from_data(
                    ctx,
                    vk::BufferUsageFlags::INDEX_BUFFER,
                    bytemuck::cast_slice(&indices),
                ));

                index_counts.push(indices.len() as u32);
            }
        }

        Self {
            vertex_buffers,
            index_buffers,
            index_counts,
            images,
            samplers,
            object_buffer,
            material_buffer,
            primitives,
            vertices_count,
            triangles_count,
        }
    }
}
