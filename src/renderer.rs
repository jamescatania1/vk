use std::{
    ffi::CString,
    time::{Duration, Instant},
};

use ash::vk::{self};
use bytemuck::{Pod, Zeroable};
use vk_mem::Alloc;
use winit::window::Window;

use crate::{
    camera::{CASCADES, Camera},
    config::{Config, DEBUG_VIEW_NAMES, DEBUG_VIEWS},
    input::Input,
    scene::{self, SceneResources},
    screen::{ExtentExt, ScreenResources},
    shaders::Shaders,
    utils::{
        buffer::Buffer,
        context::VkCtx,
        image::{Image, image},
    },
};

const PASS_SHADOW: usize = 0;
const PASS_PRIMARY: usize = 1;
const PASS_AO: usize = 2;
const PASS_AO_REPROJECT: usize = 3;
const PASS_DEFERRED: usize = 4;
const PASS_POSTFX: usize = 5;
const PASS_UI: usize = 6;

const PASS_COUNT: usize = 7;
const PASS_NAMES: [&'static str; PASS_COUNT] = [
    "Shadow",
    "Primary",
    "AO",
    "AO Reproject",
    "Deferred",
    "PostFX",
    "UI",
];

const MAX_FRAMES_IN_FLIGHT: u32 = 2;
const FRAME_ACC_ALPHA: f64 = 1.0 / 30.0;
const SWAPCHAIN_FORMAT: vk::Format = vk::Format::B8G8R8A8_SRGB;
const DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;
const SHADOWMAP_FORMAT: vk::Format = vk::Format::D32_SFLOAT;
pub const SHADOWMAP_SIZE: u32 = 2048;

pub struct Renderer {
    pub window: Window,
    ctx: VkCtx,
    query_pool: vk::QueryPool,

    config: Config,
    screen: ScreenResources,
    sampler_linear: vk::Sampler,
    sampler_nearest: vk::Sampler,
    shadowmap: Image,
    scene: SceneResources,
    frame_data: Vec<Buffer>,
    frame_id: u64,
    prev_fixed_time: Instant,
    prev_frame_time: Instant,
    avg_delta_time: Duration,
    avg_pass_times: [Duration; PASS_COUNT],
    avg_gpu_time: Duration,
    pub input: Input,
    camera: Camera,
    pub cursor_locked: bool,

    pub imgui: imgui::Context,
    pub imgui_platform: imgui_winit_support::WinitPlatform,
    imgui_renderer: imgui_rs_vulkan_renderer::Renderer,

    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    shadow_pipeline_layout: vk::PipelineLayout,
    shadow_pipeline: vk::Pipeline,
    ao_pipeline_layout: vk::PipelineLayout,
    ao_pipeline: vk::Pipeline,
    ao_reproject_pipeline_layout: vk::PipelineLayout,
    ao_reproject_pipeline: vk::Pipeline,
    deferred_pipeline_layout: vk::PipelineLayout,
    deferred_pipeline: vk::Pipeline,
    postfx_pipeline_layout: vk::PipelineLayout,
    postfx_pipeline: vk::Pipeline,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_set: vk::DescriptorSet,
    image_index: usize,
    frame_index: usize,
}

#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct FrameData {
    pub view_proj: [[f32; 4]; 4],
    pub view: [[f32; 4]; 4],
    pub proj: [[f32; 4]; 4],
    pub inv_view: [[f32; 4]; 4],
    pub inv_proj: [[f32; 4]; 4],
    pub inv_view_proj: [[f32; 4]; 4],
    pub prev_view_proj: [[f32; 4]; 4],
    pub camera_pos: [f32; 3],
    pub size: [u32; 2],
    pub texel_size: [f32; 2],
    pub size_half: [u32; 2],
    pub texel_size_half: [f32; 2],
    pub ndc_view_pixel_size: [f32; 2],
    pub frame_id: u32,
    pub light_dir: [f32; 3],
    pub cascades: [Cascade; CASCADES],
    pub ao_slices: u32,
    pub ao_samples: u32,
    pub ao_radius: f32,
    pub ao_falloff_range: f32,
    pub ao_sample_distribution_power: f32,
    pub ao_thin_occluder_compensation: f32,
    pub ao_final_value_power: f32,
    pub debug_view: u32,
}

#[derive(Debug, Default, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct Cascade {
    pub view_proj: [[f32; 4]; 4],
    pub texel_size: [f32; 2],
    pub near: f32,
    pub far: f32,
}

#[derive(Debug, Clone, Copy, Default, Zeroable, Pod)]
#[repr(C)]
pub struct PrimaryPushConstants {
    pub frame_ptr: vk::DeviceAddress,
    pub objects_ptr: vk::DeviceAddress,
    pub materials_ptr: vk::DeviceAddress,
    pub object_id: u32,
    pub material_id: u32,
}

#[derive(Debug, Clone, Copy, Default, Zeroable, Pod)]
#[repr(C)]
pub struct ShadowPushConstants {
    pub frame_ptr: vk::DeviceAddress,
    pub objects_ptr: vk::DeviceAddress,
    pub materials_ptr: vk::DeviceAddress,
    pub object_id: u32,
    pub material_id: u32,
    pub cascade_index: u32,
    pub _pad: u32,
}

#[derive(Debug, Clone, Copy, Default, Zeroable, Pod)]
#[repr(C)]
pub struct BasicPushConstants {
    pub frame_ptr: vk::DeviceAddress,
}

impl Renderer {
    pub fn new(window: Window) -> Self {
        let config = Config::default();

        let ctx = VkCtx::new(&window, MAX_FRAMES_IN_FLIGHT);

        let mut screen = ScreenResources::new(&ctx, &config);

        let frame_data = (0..MAX_FRAMES_IN_FLIGHT)
            .map(|_| {
                Buffer::new(
                    &ctx,
                    std::mem::size_of::<FrameData>() as u64,
                    vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                        | vk::BufferUsageFlags::STORAGE_BUFFER,
                    vk_mem::MemoryUsage::Auto,
                    vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
                        | vk_mem::AllocationCreateFlags::MAPPED,
                )
            })
            .collect::<Vec<_>>();

        let sampler_linear = unsafe {
            ctx.device
                .create_sampler(
                    &vk::SamplerCreateInfo::default()
                        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                        .anisotropy_enable(false)
                        .mag_filter(vk::Filter::LINEAR)
                        .min_filter(vk::Filter::LINEAR),
                    None,
                )
                .unwrap()
        };
        let sampler_nearest = unsafe {
            ctx.device
                .create_sampler(
                    &vk::SamplerCreateInfo::default()
                        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                        .anisotropy_enable(false)
                        .mag_filter(vk::Filter::NEAREST)
                        .min_filter(vk::Filter::NEAREST),
                    None,
                )
                .unwrap()
        };

        let mut shadowmap = image()
            .extent_2d_array(
                vk::Extent2D {
                    width: SHADOWMAP_SIZE,
                    height: SHADOWMAP_SIZE,
                },
                CASCADES as u32,
            )
            .format(SHADOWMAP_FORMAT)
            .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED)
            .create(&ctx, vk_mem::AllocationCreateFlags::DEDICATED_MEMORY);

        let query_pool = unsafe {
            ctx.device
                .create_query_pool(
                    &vk::QueryPoolCreateInfo::default()
                        .query_type(vk::QueryType::TIMESTAMP)
                        .query_count(MAX_FRAMES_IN_FLIGHT * (PASS_COUNT as u32 + 1)),
                    None,
                )
                .unwrap()
        };

        let mut imgui = imgui::Context::create();
        imgui.set_ini_filename(None);
        let mut imgui_platform = imgui_winit_support::WinitPlatform::new(&mut imgui);
        imgui_platform.attach_window(
            imgui.io_mut(),
            &window,
            imgui_winit_support::HiDpiMode::Default,
        );
        let imgui_renderer = imgui_rs_vulkan_renderer::Renderer::with_default_allocator(
            &ctx.instance,
            ctx.physical_device,
            ctx.device.clone(),
            ctx.queue,
            ctx.command_pool,
            imgui_rs_vulkan_renderer::DynamicRendering {
                color_attachment_format: SWAPCHAIN_FORMAT,
                depth_attachment_format: Some(DEPTH_FORMAT),
            },
            &mut imgui,
            Some(imgui_rs_vulkan_renderer::Options {
                in_flight_frames: MAX_FRAMES_IN_FLIGHT as usize,
                enable_depth_test: true,
                enable_depth_write: true,
                sample_count: vk::SampleCountFlags::TYPE_1,
                ..Default::default()
            }),
        )
        .unwrap();

        let mut scene = SceneResources::create(&ctx);

        let descriptor_pool = unsafe {
            ctx.device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(1)
                        .pool_sizes(&[
                            vk::DescriptorPoolSize::default()
                                .ty(vk::DescriptorType::SAMPLED_IMAGE)
                                .descriptor_count(scene.images.len() as u32),
                            vk::DescriptorPoolSize::default()
                                .ty(vk::DescriptorType::SAMPLER)
                                .descriptor_count(scene.samplers.len() as u32),
                            vk::DescriptorPoolSize::default()
                                .ty(vk::DescriptorType::SAMPLER)
                                .descriptor_count(1),
                            vk::DescriptorPoolSize::default()
                                .ty(vk::DescriptorType::SAMPLER)
                                .descriptor_count(1),
                            vk::DescriptorPoolSize::default()
                                .ty(vk::DescriptorType::SAMPLED_IMAGE)
                                .descriptor_count(CASCADES as u32),
                            vk::DescriptorPoolSize::default()
                                .ty(vk::DescriptorType::SAMPLED_IMAGE)
                                .descriptor_count(1),
                            vk::DescriptorPoolSize::default()
                                .ty(vk::DescriptorType::SAMPLED_IMAGE)
                                .descriptor_count(1),
                            vk::DescriptorPoolSize::default()
                                .ty(vk::DescriptorType::STORAGE_IMAGE)
                                .descriptor_count(1),
                            vk::DescriptorPoolSize::default()
                                .ty(vk::DescriptorType::SAMPLED_IMAGE)
                                .descriptor_count(1),
                            vk::DescriptorPoolSize::default()
                                .ty(vk::DescriptorType::STORAGE_IMAGE)
                                .descriptor_count(1),
                            vk::DescriptorPoolSize::default()
                                .ty(vk::DescriptorType::SAMPLED_IMAGE)
                                .descriptor_count(1),
                            vk::DescriptorPoolSize::default()
                                .ty(vk::DescriptorType::STORAGE_IMAGE)
                                .descriptor_count(2),
                            vk::DescriptorPoolSize::default()
                                .ty(vk::DescriptorType::SAMPLED_IMAGE)
                                .descriptor_count(2),
                        ]),
                    None,
                )
                .unwrap()
        };
        let descriptor_set_layout = unsafe {
            ctx.device
                .create_descriptor_set_layout(
                    &&vk::DescriptorSetLayoutCreateInfo::default().bindings(&[
                        vk::DescriptorSetLayoutBinding::default()
                            .binding(0)
                            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                            .descriptor_count(scene.images.len() as u32)
                            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
                        vk::DescriptorSetLayoutBinding::default()
                            .binding(1)
                            .descriptor_type(vk::DescriptorType::SAMPLER)
                            .descriptor_count(scene.samplers.len() as u32)
                            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
                        vk::DescriptorSetLayoutBinding::default()
                            .binding(2)
                            .descriptor_type(vk::DescriptorType::SAMPLER)
                            .descriptor_count(1)
                            .stage_flags(
                                vk::ShaderStageFlags::FRAGMENT | vk::ShaderStageFlags::COMPUTE,
                            ),
                        vk::DescriptorSetLayoutBinding::default()
                            .binding(3)
                            .descriptor_type(vk::DescriptorType::SAMPLER)
                            .descriptor_count(1)
                            .stage_flags(
                                vk::ShaderStageFlags::FRAGMENT | vk::ShaderStageFlags::COMPUTE,
                            ),
                        vk::DescriptorSetLayoutBinding::default()
                            .binding(4)
                            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                            .descriptor_count(CASCADES as u32)
                            .stage_flags(
                                vk::ShaderStageFlags::FRAGMENT | vk::ShaderStageFlags::COMPUTE,
                            ),
                        vk::DescriptorSetLayoutBinding::default()
                            .binding(5)
                            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                            .descriptor_count(1)
                            .stage_flags(
                                vk::ShaderStageFlags::FRAGMENT | vk::ShaderStageFlags::COMPUTE,
                            ),
                        vk::DescriptorSetLayoutBinding::default()
                            .binding(6)
                            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                            .descriptor_count(1)
                            .stage_flags(
                                vk::ShaderStageFlags::FRAGMENT | vk::ShaderStageFlags::COMPUTE,
                            ),
                        vk::DescriptorSetLayoutBinding::default()
                            .binding(7)
                            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                            .descriptor_count(1)
                            .stage_flags(
                                vk::ShaderStageFlags::FRAGMENT | vk::ShaderStageFlags::COMPUTE,
                            ),
                        vk::DescriptorSetLayoutBinding::default()
                            .binding(8)
                            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                            .descriptor_count(1)
                            .stage_flags(
                                vk::ShaderStageFlags::FRAGMENT | vk::ShaderStageFlags::COMPUTE,
                            ),
                        vk::DescriptorSetLayoutBinding::default()
                            .binding(9)
                            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                            .descriptor_count(1)
                            .stage_flags(
                                vk::ShaderStageFlags::FRAGMENT | vk::ShaderStageFlags::COMPUTE,
                            ),
                        vk::DescriptorSetLayoutBinding::default()
                            .binding(10)
                            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                            .descriptor_count(1)
                            .stage_flags(
                                vk::ShaderStageFlags::FRAGMENT | vk::ShaderStageFlags::COMPUTE,
                            ),
                        vk::DescriptorSetLayoutBinding::default()
                            .binding(11)
                            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                            .descriptor_count(2)
                            .stage_flags(
                                vk::ShaderStageFlags::FRAGMENT | vk::ShaderStageFlags::COMPUTE,
                            ),
                        vk::DescriptorSetLayoutBinding::default()
                            .binding(12)
                            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                            .descriptor_count(2)
                            .stage_flags(
                                vk::ShaderStageFlags::FRAGMENT | vk::ShaderStageFlags::COMPUTE,
                            ),
                    ]),
                    None,
                )
                .unwrap()
        };
        let descriptor_set = unsafe {
            ctx.device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(descriptor_pool)
                        .set_layouts(&[descriptor_set_layout]),
                )
                .unwrap()[0]
        };
        let scene_image_infos = scene
            .images
            .iter_mut()
            .map(|img| img.info_default(&ctx, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL))
            .collect::<Vec<_>>();
        let scene_sampler_infos = scene
            .samplers
            .iter()
            .map(|sampler| vk::DescriptorImageInfo::default().sampler(*sampler))
            .collect::<Vec<_>>();
        let sampler_linear_info = vk::DescriptorImageInfo::default().sampler(sampler_linear);
        let sampler_nearest_info = vk::DescriptorImageInfo::default().sampler(sampler_nearest);
        let shadowmap_infos = (0..CASCADES)
            .map(|i| {
                shadowmap.info_array_layer(&ctx, vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL, i as u32)
            })
            .collect::<Vec<_>>();
        unsafe {
            ctx.device.update_descriptor_sets(
                &[
                    vk::WriteDescriptorSet::default()
                        .dst_set(descriptor_set)
                        .dst_binding(0)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                        .image_info(&scene_image_infos),
                    vk::WriteDescriptorSet::default()
                        .dst_set(descriptor_set)
                        .dst_binding(1)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::SAMPLER)
                        .image_info(&scene_sampler_infos),
                    vk::WriteDescriptorSet::default()
                        .dst_set(descriptor_set)
                        .dst_binding(2)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::SAMPLER)
                        .image_info(&[sampler_linear_info]),
                    vk::WriteDescriptorSet::default()
                        .dst_set(descriptor_set)
                        .dst_binding(3)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::SAMPLER)
                        .image_info(&[sampler_nearest_info]),
                    vk::WriteDescriptorSet::default()
                        .dst_set(descriptor_set)
                        .dst_binding(4)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                        .image_info(&shadowmap_infos),
                ],
                &[],
            )
        };
        screen.update_descriptors(&ctx, descriptor_set);

        let shaders = Shaders::new(&ctx.device);

        let pipeline_layout = unsafe {
            ctx.device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .push_constant_ranges(&[vk::PushConstantRange::default()
                            .stage_flags(
                                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                            )
                            .size(std::mem::size_of::<PrimaryPushConstants>() as u32)])
                        .set_layouts(&[descriptor_set_layout]),
                    None,
                )
                .unwrap()
        };

        let pipeline = unsafe {
            ctx.device
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::GraphicsPipelineCreateInfo::default()
                        .push_next(
                            &mut vk::PipelineRenderingCreateInfo::default()
                                .color_attachment_formats(&[vk::Format::R32G32B32A32_UINT])
                                .depth_attachment_format(DEPTH_FORMAT),
                        )
                        .stages(&[
                            vk::PipelineShaderStageCreateInfo::default()
                                .stage(vk::ShaderStageFlags::VERTEX)
                                .name(&CString::new("main").unwrap())
                                .module(shaders.main.vertex),
                            vk::PipelineShaderStageCreateInfo::default()
                                .stage(vk::ShaderStageFlags::FRAGMENT)
                                .name(&CString::new("main").unwrap())
                                .module(shaders.main.fragment),
                        ])
                        .vertex_input_state(
                            &vk::PipelineVertexInputStateCreateInfo::default()
                                .vertex_binding_descriptions(&[
                                    vk::VertexInputBindingDescription::default()
                                        .binding(0)
                                        .stride(std::mem::size_of::<scene::VertexData>() as u32)
                                        .input_rate(vk::VertexInputRate::VERTEX),
                                ])
                                .vertex_attribute_descriptions(&[
                                    vk::VertexInputAttributeDescription::default()
                                        .location(0)
                                        .binding(0)
                                        .format(vk::Format::R32G32B32_SFLOAT)
                                        .offset(std::mem::offset_of!(scene::VertexData, position)
                                            as u32),
                                    vk::VertexInputAttributeDescription::default()
                                        .location(1)
                                        .binding(0)
                                        .format(vk::Format::R32G32B32_SFLOAT)
                                        .offset(
                                            std::mem::offset_of!(scene::VertexData, normal) as u32
                                        ),
                                    vk::VertexInputAttributeDescription::default()
                                        .location(2)
                                        .binding(0)
                                        .format(vk::Format::R32G32B32A32_SFLOAT)
                                        .offset(
                                            std::mem::offset_of!(scene::VertexData, tangent) as u32
                                        ),
                                    vk::VertexInputAttributeDescription::default()
                                        .location(3)
                                        .binding(0)
                                        .format(vk::Format::R32G32B32_SFLOAT)
                                        .offset(
                                            std::mem::offset_of!(scene::VertexData, color) as u32
                                        ),
                                    vk::VertexInputAttributeDescription::default()
                                        .location(4)
                                        .binding(0)
                                        .format(vk::Format::R32G32_SFLOAT)
                                        .offset(std::mem::offset_of!(scene::VertexData, uv) as u32),
                                ]),
                        )
                        .input_assembly_state(
                            &vk::PipelineInputAssemblyStateCreateInfo::default()
                                .topology(vk::PrimitiveTopology::TRIANGLE_LIST),
                        )
                        .viewport_state(
                            &vk::PipelineViewportStateCreateInfo::default()
                                .viewport_count(1)
                                .scissor_count(1),
                        )
                        .rasterization_state(
                            &vk::PipelineRasterizationStateCreateInfo::default()
                                .cull_mode(vk::CullModeFlags::BACK)
                                .line_width(1.0),
                        )
                        .multisample_state(
                            &vk::PipelineMultisampleStateCreateInfo::default()
                                .rasterization_samples(vk::SampleCountFlags::TYPE_1),
                        )
                        .depth_stencil_state(
                            &vk::PipelineDepthStencilStateCreateInfo::default()
                                .depth_test_enable(true)
                                .depth_write_enable(true)
                                .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL),
                        )
                        .color_blend_state(
                            &vk::PipelineColorBlendStateCreateInfo::default()
                                .attachments(&[vk::PipelineColorBlendAttachmentState::default()
                                    .color_write_mask(vk::ColorComponentFlags::RGBA)]),
                        )
                        .dynamic_state(
                            &vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&[
                                vk::DynamicState::VIEWPORT,
                                vk::DynamicState::SCISSOR,
                            ]),
                        )
                        .layout(pipeline_layout)],
                    None,
                )
                .unwrap()[0]
        };

        let shadow_pipeline_layout = unsafe {
            ctx.device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&[
                        vk::PushConstantRange::default()
                            .stage_flags(
                                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                            )
                            .size(std::mem::size_of::<ShadowPushConstants>() as u32),
                    ]),
                    None,
                )
                .unwrap()
        };

        let shadow_pipeline = unsafe {
            ctx.device
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::GraphicsPipelineCreateInfo::default()
                        .push_next(
                            &mut vk::PipelineRenderingCreateInfo::default()
                                .color_attachment_formats(&[])
                                .depth_attachment_format(SHADOWMAP_FORMAT),
                        )
                        .stages(&[
                            vk::PipelineShaderStageCreateInfo::default()
                                .stage(vk::ShaderStageFlags::VERTEX)
                                .name(&CString::new("main").unwrap())
                                .module(shaders.shadow.vertex),
                            vk::PipelineShaderStageCreateInfo::default()
                                .stage(vk::ShaderStageFlags::FRAGMENT)
                                .name(&CString::new("main").unwrap())
                                .module(shaders.shadow.fragment),
                        ])
                        .vertex_input_state(
                            &vk::PipelineVertexInputStateCreateInfo::default()
                                .vertex_binding_descriptions(&[
                                    vk::VertexInputBindingDescription::default()
                                        .binding(0)
                                        .stride(std::mem::size_of::<scene::VertexData>() as u32)
                                        .input_rate(vk::VertexInputRate::VERTEX),
                                ])
                                .vertex_attribute_descriptions(&[
                                    vk::VertexInputAttributeDescription::default()
                                        .location(0)
                                        .binding(0)
                                        .format(vk::Format::R32G32B32_SFLOAT)
                                        .offset(std::mem::offset_of!(scene::VertexData, position)
                                            as u32),
                                ]),
                        )
                        .input_assembly_state(
                            &vk::PipelineInputAssemblyStateCreateInfo::default()
                                .topology(vk::PrimitiveTopology::TRIANGLE_LIST),
                        )
                        .viewport_state(
                            &vk::PipelineViewportStateCreateInfo::default()
                                .viewport_count(1)
                                .scissor_count(1),
                        )
                        .rasterization_state(
                            &vk::PipelineRasterizationStateCreateInfo::default()
                                .cull_mode(vk::CullModeFlags::BACK)
                                .line_width(1.0),
                        )
                        .multisample_state(
                            &vk::PipelineMultisampleStateCreateInfo::default()
                                .rasterization_samples(vk::SampleCountFlags::TYPE_1),
                        )
                        .depth_stencil_state(
                            &vk::PipelineDepthStencilStateCreateInfo::default()
                                .depth_test_enable(true)
                                .depth_write_enable(true)
                                .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL),
                        )
                        .color_blend_state(&vk::PipelineColorBlendStateCreateInfo::default())
                        .dynamic_state(
                            &vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&[
                                vk::DynamicState::VIEWPORT,
                                vk::DynamicState::SCISSOR,
                            ]),
                        )
                        .layout(shadow_pipeline_layout)],
                    None,
                )
                .unwrap()[0]
        };

        let ao_pipeline_layout = unsafe {
            ctx.device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .push_constant_ranges(&[vk::PushConstantRange::default()
                            .stage_flags(vk::ShaderStageFlags::COMPUTE)
                            .size(std::mem::size_of::<BasicPushConstants>() as u32)])
                        .set_layouts(&[descriptor_set_layout]),
                    None,
                )
                .unwrap()
        };
        let ao_pipeline = unsafe {
            ctx.device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::ComputePipelineCreateInfo::default()
                        .stage(
                            vk::PipelineShaderStageCreateInfo::default()
                                .stage(vk::ShaderStageFlags::COMPUTE)
                                .name(&CString::new("main").unwrap())
                                .module(shaders.ao.main),
                        )
                        .layout(ao_pipeline_layout)],
                    None,
                )
                .unwrap()[0]
        };

        let ao_reproject_pipeline_layout = unsafe {
            ctx.device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .push_constant_ranges(&[vk::PushConstantRange::default()
                            .stage_flags(vk::ShaderStageFlags::COMPUTE)
                            .size(std::mem::size_of::<BasicPushConstants>() as u32)])
                        .set_layouts(&[descriptor_set_layout]),
                    None,
                )
                .unwrap()
        };
        let ao_reproject_pipeline = unsafe {
            ctx.device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::ComputePipelineCreateInfo::default()
                        .stage(
                            vk::PipelineShaderStageCreateInfo::default()
                                .stage(vk::ShaderStageFlags::COMPUTE)
                                .name(&CString::new("main").unwrap())
                                .module(shaders.ao_reproject.main),
                        )
                        .layout(ao_reproject_pipeline_layout)],
                    None,
                )
                .unwrap()[0]
        };

        let deferred_pipeline_layout = unsafe {
            ctx.device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .push_constant_ranges(&[vk::PushConstantRange::default()
                            .stage_flags(vk::ShaderStageFlags::COMPUTE)
                            .size(std::mem::size_of::<BasicPushConstants>() as u32)])
                        .set_layouts(&[descriptor_set_layout]),
                    None,
                )
                .unwrap()
        };
        let deferred_pipeline = unsafe {
            ctx.device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::ComputePipelineCreateInfo::default()
                        .stage(
                            vk::PipelineShaderStageCreateInfo::default()
                                .stage(vk::ShaderStageFlags::COMPUTE)
                                .name(&CString::new("main").unwrap())
                                .module(shaders.deferred.main),
                        )
                        .layout(deferred_pipeline_layout)],
                    None,
                )
                .unwrap()[0]
        };

        let postfx_pipeline_layout = unsafe {
            ctx.device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .push_constant_ranges(&[vk::PushConstantRange::default()
                            .stage_flags(
                                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                            )
                            .size(std::mem::size_of::<BasicPushConstants>() as u32)])
                        .set_layouts(&[descriptor_set_layout]),
                    None,
                )
                .unwrap()
        };

        let postfx_pipeline = unsafe {
            ctx.device
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::GraphicsPipelineCreateInfo::default()
                        .push_next(
                            &mut vk::PipelineRenderingCreateInfo::default()
                                .color_attachment_formats(&[SWAPCHAIN_FORMAT]),
                        )
                        .stages(&[
                            vk::PipelineShaderStageCreateInfo::default()
                                .stage(vk::ShaderStageFlags::VERTEX)
                                .name(&CString::new("main").unwrap())
                                .module(shaders.postfx.vertex),
                            vk::PipelineShaderStageCreateInfo::default()
                                .stage(vk::ShaderStageFlags::FRAGMENT)
                                .name(&CString::new("main").unwrap())
                                .module(shaders.postfx.fragment),
                        ])
                        .vertex_input_state(&vk::PipelineVertexInputStateCreateInfo::default())
                        .input_assembly_state(
                            &vk::PipelineInputAssemblyStateCreateInfo::default()
                                .topology(vk::PrimitiveTopology::TRIANGLE_LIST),
                        )
                        .viewport_state(
                            &vk::PipelineViewportStateCreateInfo::default()
                                .viewport_count(1)
                                .scissor_count(1),
                        )
                        .rasterization_state(
                            &vk::PipelineRasterizationStateCreateInfo::default().line_width(1.0),
                        )
                        .multisample_state(
                            &vk::PipelineMultisampleStateCreateInfo::default()
                                .rasterization_samples(vk::SampleCountFlags::TYPE_1),
                        )
                        .depth_stencil_state(&vk::PipelineDepthStencilStateCreateInfo::default())
                        .color_blend_state(
                            &vk::PipelineColorBlendStateCreateInfo::default()
                                .attachments(&[vk::PipelineColorBlendAttachmentState::default()
                                    .color_write_mask(vk::ColorComponentFlags::RGBA)]),
                        )
                        .dynamic_state(
                            &vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&[
                                vk::DynamicState::VIEWPORT,
                                vk::DynamicState::SCISSOR,
                            ]),
                        )
                        .layout(postfx_pipeline_layout)],
                    None,
                )
                .unwrap()[0]
        };

        Self {
            window,
            ctx,
            screen,
            shadowmap,
            sampler_linear,
            sampler_nearest,
            frame_data,
            scene,
            frame_id: 0,
            prev_frame_time: Instant::now(),
            prev_fixed_time: Instant::now(),
            avg_delta_time: Duration::ZERO,
            avg_gpu_time: Duration::ZERO,
            avg_pass_times: [Duration::ZERO; PASS_COUNT],
            input: Default::default(),
            camera: Camera::new(),
            config,
            cursor_locked: false,
            pipeline_layout,
            pipeline,
            shadow_pipeline_layout,
            shadow_pipeline,
            ao_reproject_pipeline_layout,
            ao_reproject_pipeline,
            ao_pipeline_layout,
            ao_pipeline,
            deferred_pipeline_layout,
            deferred_pipeline,
            postfx_pipeline_layout,
            postfx_pipeline,
            imgui,
            imgui_platform,
            imgui_renderer,
            query_pool,
            descriptor_pool,
            descriptor_set_layout,
            descriptor_set,
            image_index: 0,
            frame_index: 0,
        }
    }

    pub fn on_resize(&mut self) {
        self.screen.recreate = true;
    }

    pub fn frame(&mut self) {
        let time = Instant::now();
        let delta_time = time - self.prev_frame_time;
        self.prev_frame_time = time;

        unsafe {
            self.ctx
                .device
                .wait_for_fences(&[self.ctx.frame_fences[self.frame_index]], true, u64::MAX)
                .unwrap();
        };

        let query_base = self.frame_index as u32 * (PASS_COUNT as u32 + 1);

        if self.frame_id >= MAX_FRAMES_IN_FLIGHT as u64 {
            let mut timestamps = [0u64; PASS_COUNT + 1];
            unsafe {
                match self.ctx.device.get_query_pool_results(
                    self.query_pool,
                    query_base,
                    &mut timestamps,
                    vk::QueryResultFlags::TYPE_64,
                ) {
                    Ok(()) => {
                        let delta_total = Duration::from_nanos(
                            (timestamps[PASS_COUNT].wrapping_sub(timestamps[0]) as f64
                                * self.ctx.physical_device_properties.limits.timestamp_period
                                    as f64) as u64,
                        );
                        self.avg_gpu_time = self.avg_gpu_time.mul_f64(1.0 - FRAME_ACC_ALPHA)
                            + delta_total.mul_f64(FRAME_ACC_ALPHA);

                        for i in 0..PASS_COUNT {
                            let delta = Duration::from_nanos(
                                (timestamps[i + 1].wrapping_sub(timestamps[i]) as f64
                                    * self.ctx.physical_device_properties.limits.timestamp_period
                                        as f64) as u64,
                            );
                            self.avg_pass_times[i] = self.avg_pass_times[i]
                                .mul_f64(1.0 - FRAME_ACC_ALPHA)
                                + delta.mul_f64(FRAME_ACC_ALPHA);
                        }
                    }
                    Err(vk::Result::NOT_READY) => {}
                    Err(err) => panic!("{err:?}"),
                }
            };
        }

        if self.screen.recreate {
            self.screen
                .recreate(&self.ctx, &self.config, self.descriptor_set);
        }

        let (next_image_index, _suboptimal) = unsafe {
            match self.screen.swapchain_loader.acquire_next_image(
                self.screen.swapchain,
                u64::MAX,
                self.ctx.image_acquired_semaphores[self.frame_index],
                vk::Fence::null(),
            ) {
                Ok(res) => res,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    self.screen.recreate = true;
                    return;
                }
                Err(err) => {
                    panic!("{err:?}");
                }
            }
        };

        unsafe {
            self.ctx
                .device
                .reset_fences(&[self.ctx.frame_fences[self.frame_index]])
                .unwrap()
        }
        self.image_index = next_image_index as usize;

        if !self.cursor_locked && self.input.key_pressed(winit::keyboard::KeyCode::Space) {
            self.cursor_locked = true;
            self.window
                .set_cursor_grab(winit::window::CursorGrabMode::Locked)
                .or_else(|_| {
                    self.window
                        .set_cursor_grab(winit::window::CursorGrabMode::Confined)
                })
                .unwrap();
            self.window.set_cursor_visible(false);
        } else if self.cursor_locked && self.input.key_pressed(winit::keyboard::KeyCode::Space) {
            self.cursor_locked = false;

            self.window
                .set_cursor_grab(winit::window::CursorGrabMode::None)
                .unwrap();
            self.window.set_cursor_visible(true);
        }

        let az = self.config.sun_azimuth.to_radians() as f32;
        let alt = self.config.sun_altitude.to_radians() as f32;
        let sun_dir = glam::vec3(alt.cos() * az.sin(), alt.cos() * az.cos(), alt.sin()).normalize();

        self.camera.update(
            glam::uvec2(
                self.screen.render_size.width,
                self.screen.render_size.height,
            ),
            &delta_time,
            &self.input,
            self.cursor_locked,
            sun_dir,
            self.config.cascade_lambda,
            &self.scene,
            self.screen.texel_size,
        );

        self.input.update();

        self.frame_data[self.frame_index].write(
            &self.ctx,
            bytemuck::bytes_of(&FrameData {
                view_proj: self.camera.view_proj.to_cols_array_2d(),
                view: self.camera.view.to_cols_array_2d(),
                proj: self.camera.proj.to_cols_array_2d(),
                inv_view: self.camera.inv_view.to_cols_array_2d(),
                inv_proj: self.camera.inv_proj.to_cols_array_2d(),
                inv_view_proj: self.camera.inv_view_proj.to_cols_array_2d(),
                prev_view_proj: self.camera.prev_view_proj.to_cols_array_2d(),
                camera_pos: self.camera.position.to_array(),
                size: self.screen.render_size.as_uvec2().to_array(),
                texel_size: self.screen.texel_size.to_array(),
                size_half: self.screen.render_size_half.as_uvec2().to_array(),
                texel_size_half: self.screen.texel_size_half.to_array(),
                ndc_view_pixel_size: self.camera.ndc_view_pixel_size.to_array(),
                frame_id: self.frame_id as u32,
                light_dir: sun_dir.to_array(),
                cascades: self.camera.cascades.map(|cascade| Cascade {
                    near: cascade.near,
                    far: cascade.far,
                    texel_size: cascade.texel_size.to_array(),
                    view_proj: cascade.view_proj.to_cols_array_2d(),
                }),
                ao_slices: self.config.ao_slices,
                ao_samples: self.config.ao_samples,
                ao_radius: self.config.ao_radius,
                ao_falloff_range: self.config.ao_falloff_range,
                ao_sample_distribution_power: self.config.ao_sample_distribution_power,
                ao_thin_occluder_compensation: self.config.ao_thin_occluder_compensation,
                ao_final_value_power: self.config.ao_final_value_power,
                debug_view: self.config.debug_view as u32,
            }),
        );
        self.frame_data[self.frame_index].flush(&self.ctx);

        let frame_ptr = self.frame_data[self.frame_index].address(&self.ctx);
        let objects_ptr = self.scene.object_buffer.address(&self.ctx);
        let materials_ptr = self.scene.material_buffer.address(&self.ctx);

        let cb = self.ctx.command_buffers[self.frame_index];

        let swap_index_a = (self.frame_id as u32 & 1) as usize;
        let swap_index_b = ((self.frame_id as u32 + 1) & 1) as usize;

        unsafe {
            self.ctx
                .device
                .reset_command_buffer(cb, vk::CommandBufferResetFlags::empty())
                .unwrap();
            self.ctx
                .device
                .begin_command_buffer(
                    cb,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .unwrap();

            self.ctx.device.cmd_reset_query_pool(
                cb,
                self.query_pool,
                query_base,
                PASS_COUNT as u32 + 1,
            );

            // shadow pass
            self.ctx.device.cmd_write_timestamp2(
                cb,
                vk::PipelineStageFlags2::TOP_OF_PIPE,
                self.query_pool,
                query_base + PASS_SHADOW as u32,
            );
            for cascade in 0..CASCADES {
                self.ctx.device.cmd_pipeline_barrier2(
                    cb,
                    &vk::DependencyInfo::default().image_memory_barriers(&[
                        vk::ImageMemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::NONE)
                            .dst_stage_mask(
                                vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                                    | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                            )
                            .src_access_mask(vk::AccessFlags2::NONE)
                            .dst_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
                            .old_layout(vk::ImageLayout::UNDEFINED)
                            .new_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                            .image(self.shadowmap.image)
                            .subresource_range(
                                vk::ImageSubresourceRange::default()
                                    .aspect_mask(vk::ImageAspectFlags::DEPTH)
                                    .level_count(1)
                                    .base_array_layer(cascade as u32)
                                    .layer_count(1),
                            ),
                    ]),
                );

                self.ctx.device.cmd_begin_rendering(
                    cb,
                    &vk::RenderingInfo::default()
                        .render_area(vk::Rect2D::default().extent(vk::Extent2D {
                            width: SHADOWMAP_SIZE,
                            height: SHADOWMAP_SIZE,
                        }))
                        .layer_count(1)
                        .color_attachments(&[])
                        .depth_attachment(
                            &vk::RenderingAttachmentInfo::default()
                                .image_view(
                                    self.shadowmap.view_array_layer(&self.ctx, cascade as u32),
                                )
                                .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                                .load_op(vk::AttachmentLoadOp::CLEAR)
                                .store_op(vk::AttachmentStoreOp::STORE)
                                .clear_value(vk::ClearValue {
                                    depth_stencil: vk::ClearDepthStencilValue {
                                        depth: 1.0,
                                        stencil: 0,
                                    },
                                }),
                        ),
                );
                self.ctx.device.cmd_set_viewport(
                    cb,
                    0,
                    &[vk::Viewport {
                        width: SHADOWMAP_SIZE as f32,
                        height: SHADOWMAP_SIZE as f32,
                        x: 0.0,
                        y: 0.0,
                        min_depth: 0.0,
                        max_depth: 1.0,
                    }],
                );
                self.ctx.device.cmd_set_scissor(
                    cb,
                    0,
                    &[vk::Rect2D {
                        extent: vk::Extent2D {
                            width: SHADOWMAP_SIZE,
                            height: SHADOWMAP_SIZE,
                        },
                        offset: vk::Offset2D { x: 0, y: 0 },
                    }],
                );
                self.ctx.device.cmd_bind_pipeline(
                    cb,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.shadow_pipeline,
                );
                for i in 0..self.scene.vertex_buffers.len() {
                    self.ctx.device.cmd_bind_vertex_buffers(
                        cb,
                        0,
                        &[self.scene.vertex_buffers[i].buffer],
                        &[0],
                    );
                    self.ctx.device.cmd_bind_index_buffer(
                        cb,
                        self.scene.index_buffers[i].buffer,
                        0,
                        vk::IndexType::UINT32,
                    );
                    self.ctx.device.cmd_push_constants(
                        cb,
                        self.shadow_pipeline_layout,
                        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                        0,
                        bytemuck::bytes_of(&ShadowPushConstants {
                            frame_ptr,
                            objects_ptr,
                            materials_ptr,
                            object_id: self.scene.primitives[i].object_id,
                            material_id: self.scene.primitives[i].material_id,
                            cascade_index: cascade as u32,
                            _pad: 0,
                        }),
                    );
                    self.ctx
                        .device
                        .cmd_draw_indexed(cb, self.scene.index_counts[i], 1, 0, 0, 0);
                }

                self.ctx.device.cmd_end_rendering(cb);
            }

            self.ctx.device.cmd_pipeline_barrier2(
                cb,
                &vk::DependencyInfo::default().image_memory_barriers(&[
                    vk::ImageMemoryBarrier2::default()
                        .src_stage_mask(
                            vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                                | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                        )
                        .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                        .src_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
                        .dst_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ)
                        .old_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                        .new_layout(vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL)
                        .image(self.shadowmap.image)
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(vk::ImageAspectFlags::DEPTH)
                                .level_count(1)
                                .layer_count(CASCADES as u32),
                        ),
                    vk::ImageMemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                        .src_access_mask(vk::AccessFlags2::empty())
                        .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                        .dst_access_mask(
                            vk::AccessFlags2::COLOR_ATTACHMENT_READ
                                | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                        )
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::ATTACHMENT_OPTIMAL)
                        .image(self.screen.images.gbuffer.image)
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .level_count(1)
                                .layer_count(1),
                        ),
                    vk::ImageMemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS)
                        .src_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
                        .dst_stage_mask(vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS)
                        .dst_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::ATTACHMENT_OPTIMAL)
                        .image(self.screen.images.depth.image)
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(vk::ImageAspectFlags::DEPTH)
                                .level_count(1)
                                .layer_count(1),
                        ),
                ]),
            );

            // main pass
            {
                self.ctx.device.cmd_write_timestamp2(
                    cb,
                    vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
                    self.query_pool,
                    query_base + PASS_PRIMARY as u32,
                );
                self.ctx.device.cmd_begin_rendering(
                    cb,
                    &vk::RenderingInfo::default()
                        .render_area(vk::Rect2D::default().extent(self.screen.render_size))
                        .layer_count(1)
                        .color_attachments(&[vk::RenderingAttachmentInfo::default()
                            .image_view(self.screen.images.gbuffer.view_default(&self.ctx))
                            .image_layout(vk::ImageLayout::ATTACHMENT_OPTIMAL)
                            .load_op(vk::AttachmentLoadOp::CLEAR)
                            .store_op(vk::AttachmentStoreOp::STORE)
                            .clear_value(vk::ClearValue {
                                color: vk::ClearColorValue {
                                    uint32: [0, 0, 0, 0],
                                },
                            })])
                        .depth_attachment(
                            &vk::RenderingAttachmentInfo::default()
                                .image_view(self.screen.images.depth.view_default(&self.ctx))
                                .image_layout(vk::ImageLayout::ATTACHMENT_OPTIMAL)
                                .load_op(vk::AttachmentLoadOp::CLEAR)
                                .store_op(vk::AttachmentStoreOp::STORE)
                                .clear_value(vk::ClearValue {
                                    depth_stencil: vk::ClearDepthStencilValue {
                                        depth: 1.0,
                                        stencil: 0,
                                    },
                                }),
                        ),
                );
                self.ctx.device.cmd_set_viewport(
                    cb,
                    0,
                    &[vk::Viewport {
                        width: self.screen.render_size.width as f32,
                        height: self.screen.render_size.height as f32,
                        x: 0.0,
                        y: 0.0,
                        min_depth: 0.0,
                        max_depth: 1.0,
                    }],
                );
                self.ctx.device.cmd_set_scissor(
                    cb,
                    0,
                    &[vk::Rect2D {
                        extent: self.screen.render_size,
                        offset: vk::Offset2D { x: 0, y: 0 },
                    }],
                );
                self.ctx.device.cmd_bind_pipeline(
                    cb,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline,
                );
                self.ctx.device.cmd_bind_descriptor_sets(
                    cb,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline_layout,
                    0,
                    &[self.descriptor_set],
                    &[],
                );
                for i in 0..self.scene.vertex_buffers.len() {
                    self.ctx.device.cmd_bind_vertex_buffers(
                        cb,
                        0,
                        &[self.scene.vertex_buffers[i].buffer],
                        &[0],
                    );
                    self.ctx.device.cmd_bind_index_buffer(
                        cb,
                        self.scene.index_buffers[i].buffer,
                        0,
                        vk::IndexType::UINT32,
                    );
                    self.ctx.device.cmd_push_constants(
                        cb,
                        self.pipeline_layout,
                        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                        0,
                        bytemuck::bytes_of(&PrimaryPushConstants {
                            frame_ptr,
                            objects_ptr,
                            materials_ptr,
                            object_id: self.scene.primitives[i].object_id,
                            material_id: self.scene.primitives[i].material_id,
                        }),
                    );
                    self.ctx
                        .device
                        .cmd_draw_indexed(cb, self.scene.index_counts[i], 1, 0, 0, 0);
                }
                self.ctx.device.cmd_end_rendering(cb);
            }

            self.ctx.device.cmd_pipeline_barrier2(
                cb,
                &vk::DependencyInfo::default().image_memory_barriers(&[
                    vk::ImageMemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                        .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                        .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                        .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                        .old_layout(vk::ImageLayout::ATTACHMENT_OPTIMAL)
                        .new_layout(vk::ImageLayout::READ_ONLY_OPTIMAL)
                        .image(self.screen.images.gbuffer.image)
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .level_count(1)
                                .layer_count(1),
                        ),
                    vk::ImageMemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::NONE)
                        .src_access_mask(vk::AccessFlags2::NONE)
                        .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                        .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::GENERAL)
                        .image(self.screen.images.ao.image)
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .level_count(1)
                                .layer_count(1),
                        ),
                    vk::ImageMemoryBarrier2::default()
                        .src_stage_mask(
                            vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                                | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                        )
                        .src_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
                        .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                        .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                        .old_layout(vk::ImageLayout::ATTACHMENT_OPTIMAL)
                        .new_layout(vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL)
                        .image(self.screen.images.depth.image)
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(vk::ImageAspectFlags::DEPTH)
                                .level_count(1)
                                .layer_count(1),
                        ),
                ]),
            );

            // ao
            {
                self.ctx.device.cmd_write_timestamp2(
                    cb,
                    vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
                    self.query_pool,
                    query_base + PASS_AO as u32,
                );
                self.ctx.device.cmd_bind_pipeline(
                    cb,
                    vk::PipelineBindPoint::COMPUTE,
                    self.ao_pipeline,
                );
                self.ctx.device.cmd_bind_descriptor_sets(
                    cb,
                    vk::PipelineBindPoint::COMPUTE,
                    self.ao_pipeline_layout,
                    0,
                    &[self.descriptor_set],
                    &[],
                );
                self.ctx.device.cmd_push_constants(
                    cb,
                    self.ao_pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    bytemuck::bytes_of(&BasicPushConstants { frame_ptr }),
                );
                self.ctx.device.cmd_dispatch(
                    cb,
                    self.screen.render_size.width.div_ceil(8 * 2),
                    self.screen.render_size.height.div_ceil(8 * 2),
                    1,
                );
            }

            self.ctx.device.cmd_pipeline_barrier2(
                cb,
                &vk::DependencyInfo::default().image_memory_barriers(&[
                    vk::ImageMemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                        .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                        .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                        .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                        .old_layout(vk::ImageLayout::GENERAL)
                        .new_layout(vk::ImageLayout::READ_ONLY_OPTIMAL)
                        .image(self.screen.images.ao.image)
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .level_count(1)
                                .layer_count(1),
                        ),
                    vk::ImageMemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::NONE)
                        .src_access_mask(vk::AccessFlags2::NONE)
                        .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                        .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::GENERAL)
                        .image(self.screen.images.ao_history[swap_index_a].image)
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .level_count(1)
                                .layer_count(1),
                        ),
                    vk::ImageMemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::NONE)
                        .src_access_mask(vk::AccessFlags2::NONE)
                        .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                        .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::GENERAL)
                        .image(self.screen.images.ao_history[swap_index_b].image)
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .level_count(1)
                                .layer_count(1),
                        ),
                ]),
            );

            // ao reproject
            {
                self.ctx.device.cmd_write_timestamp2(
                    cb,
                    vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
                    self.query_pool,
                    query_base + PASS_AO_REPROJECT as u32,
                );
                self.ctx.device.cmd_bind_pipeline(
                    cb,
                    vk::PipelineBindPoint::COMPUTE,
                    self.ao_reproject_pipeline,
                );
                self.ctx.device.cmd_bind_descriptor_sets(
                    cb,
                    vk::PipelineBindPoint::COMPUTE,
                    self.ao_reproject_pipeline_layout,
                    0,
                    &[self.descriptor_set],
                    &[],
                );
                self.ctx.device.cmd_push_constants(
                    cb,
                    self.ao_reproject_pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    bytemuck::bytes_of(&BasicPushConstants { frame_ptr }),
                );
                self.ctx.device.cmd_dispatch(
                    cb,
                    self.screen.render_size.width.div_ceil(8 * 2),
                    self.screen.render_size.height.div_ceil(8 * 2),
                    1,
                );
            }

            self.ctx.device.cmd_pipeline_barrier2(
                cb,
                &vk::DependencyInfo::default().image_memory_barriers(&[
                    vk::ImageMemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                        .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                        .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                        .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                        .old_layout(vk::ImageLayout::GENERAL)
                        .new_layout(vk::ImageLayout::GENERAL)
                        .image(self.screen.images.ao_history[swap_index_b].image)
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .level_count(1)
                                .layer_count(1),
                        ),
                    vk::ImageMemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::NONE)
                        .src_access_mask(vk::AccessFlags2::NONE)
                        .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                        .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::GENERAL)
                        .image(self.screen.images.color_output.image)
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .level_count(1)
                                .layer_count(1),
                        ),
                ]),
            );

            // deferred
            {
                self.ctx.device.cmd_write_timestamp2(
                    cb,
                    vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
                    self.query_pool,
                    query_base + PASS_DEFERRED as u32,
                );
                self.ctx.device.cmd_bind_pipeline(
                    cb,
                    vk::PipelineBindPoint::COMPUTE,
                    self.deferred_pipeline,
                );
                self.ctx.device.cmd_bind_descriptor_sets(
                    cb,
                    vk::PipelineBindPoint::COMPUTE,
                    self.deferred_pipeline_layout,
                    0,
                    &[self.descriptor_set],
                    &[],
                );
                self.ctx.device.cmd_push_constants(
                    cb,
                    self.deferred_pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    bytemuck::bytes_of(&BasicPushConstants { frame_ptr }),
                );
                self.ctx.device.cmd_dispatch(
                    cb,
                    self.screen.render_size.width.div_ceil(8),
                    self.screen.render_size.height.div_ceil(8),
                    1,
                );
            }

            self.ctx.device.cmd_pipeline_barrier2(
                cb,
                &vk::DependencyInfo::default().image_memory_barriers(&[
                    vk::ImageMemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                        .src_access_mask(vk::AccessFlags2::empty())
                        .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                        .dst_access_mask(
                            vk::AccessFlags2::COLOR_ATTACHMENT_READ
                                | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                        )
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::ATTACHMENT_OPTIMAL)
                        .image(self.screen.swapchain_images[self.image_index])
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .level_count(1)
                                .layer_count(1),
                        ),
                    vk::ImageMemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                        .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                        .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                        .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                        .old_layout(vk::ImageLayout::GENERAL)
                        .new_layout(vk::ImageLayout::READ_ONLY_OPTIMAL)
                        .image(self.screen.images.color_output.image)
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .level_count(1)
                                .layer_count(1),
                        ),
                ]),
            );

            // post fx
            {
                self.ctx.device.cmd_write_timestamp2(
                    cb,
                    vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
                    self.query_pool,
                    query_base + PASS_POSTFX as u32,
                );
                self.ctx.device.cmd_begin_rendering(
                    cb,
                    &vk::RenderingInfo::default()
                        .render_area(vk::Rect2D::default().extent(self.screen.viewport_size))
                        .layer_count(1)
                        .color_attachments(&[vk::RenderingAttachmentInfo::default()
                            .image_view(self.screen.swapchain_views[self.image_index])
                            .image_layout(vk::ImageLayout::ATTACHMENT_OPTIMAL)
                            .load_op(vk::AttachmentLoadOp::CLEAR)
                            .store_op(vk::AttachmentStoreOp::STORE)
                            .clear_value(vk::ClearValue {
                                color: vk::ClearColorValue {
                                    float32: [0.0, 0.0, 0.0, 1.0],
                                },
                            })]),
                );
                self.ctx.device.cmd_set_viewport(
                    cb,
                    0,
                    &[vk::Viewport {
                        width: self.screen.viewport_size.width as f32,
                        height: self.screen.viewport_size.height as f32,
                        x: 0.0,
                        y: 0.0,
                        min_depth: 0.0,
                        max_depth: 1.0,
                    }],
                );
                self.ctx.device.cmd_set_scissor(
                    cb,
                    0,
                    &[vk::Rect2D {
                        extent: vk::Extent2D {
                            width: self.screen.viewport_size.width,
                            height: self.screen.viewport_size.height,
                        },
                        offset: vk::Offset2D { x: 0, y: 0 },
                    }],
                );
                self.ctx.device.cmd_bind_pipeline(
                    cb,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.postfx_pipeline,
                );
                self.ctx.device.cmd_bind_descriptor_sets(
                    cb,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.postfx_pipeline_layout,
                    0,
                    &[self.descriptor_set],
                    &[],
                );
                self.ctx.device.cmd_push_constants(
                    cb,
                    self.postfx_pipeline_layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    0,
                    bytemuck::bytes_of(&BasicPushConstants { frame_ptr }),
                );
                self.ctx.device.cmd_draw(cb, 3, 1, 0, 0);
            }

            // draw ui
            {
                self.ctx.device.cmd_write_timestamp2(
                    cb,
                    vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
                    self.query_pool,
                    query_base + PASS_UI as u32,
                );
                self.imgui_platform
                    .prepare_frame(self.imgui.io_mut(), &self.window)
                    .unwrap();
                let ui = self.imgui.frame();
                ui.window("Debug")
                    .size([300.0, 600.0], imgui::Condition::FirstUseEver)
                    .position([0.0, 0.0], imgui::Condition::FirstUseEver)
                    .build(|| {
                        ui.text(format!(
                            "Resolution: {} x {}",
                            self.screen.render_size.width, self.screen.render_size.height
                        ));
                        ui.text(format!(
                            "FPS: {:.1}",
                            1.0 / self.avg_delta_time.as_secs_f64()
                        ));
                        ui.separator();
                        ui.text(format!("Frame Time: {:#?}", self.avg_delta_time,));
                        ui.text(format!("GPU Time: {:#?}", self.avg_gpu_time,));
                        for i in 0..PASS_COUNT {
                            ui.text(format!("{}: {:#?}", PASS_NAMES[i], self.avg_pass_times[i],));
                        }
                        ui.separator();
                        if ui.slider("Render Scale", 0.125, 2.0, &mut self.config.render_scale) {
                            self.screen.recreate = true;
                        }
                        ui.separator();
                        ui.text(format!("Primitives: {}", self.scene.primitives.len()));
                        ui.text(format!("Vertices: {}", self.scene.vertices_count));
                        ui.text(format!("Triangles: {}", self.scene.triangles_count));
                        ui.separator();
                        let mut debug_view = self.config.debug_view as usize;
                        if ui.combo_simple_string("Display", &mut debug_view, &DEBUG_VIEW_NAMES) {
                            self.config.debug_view = DEBUG_VIEWS[debug_view];
                        }
                        ui.separator();
                        ui.slider("Sun Azimuth", 0.0, 360.0, &mut self.config.sun_azimuth);
                        ui.slider("Sun Altitude", 0.0, 90.0, &mut self.config.sun_altitude);
                        ui.slider("Cascade Lambda", 0.0, 1.0, &mut self.config.cascade_lambda);
                        ui.separator();
                        ui.text("Ambient Occlusion");
                        ui.slider("Slices", 1, 16, &mut self.config.ao_slices);
                        ui.slider("Samples", 1, 32, &mut self.config.ao_samples);
                        ui.slider("Radius", 0.0, 2.0, &mut self.config.ao_radius);
                        ui.slider("Falloff Range", 0.0, 1.0, &mut self.config.ao_falloff_range);
                        ui.slider(
                            "Sample Distribution Power",
                            1.0,
                            4.0,
                            &mut self.config.ao_sample_distribution_power,
                        );
                        ui.slider(
                            "Thin Occluder Compensation",
                            0.0,
                            1.0,
                            &mut self.config.ao_thin_occluder_compensation,
                        );
                        ui.slider("Power", 0.0, 4.0, &mut self.config.ao_final_value_power);
                    });
                self.imgui_platform.prepare_render(&ui, &self.window);
                self.imgui_renderer
                    .cmd_draw(cb, imgui::Context::render(&mut self.imgui))
                    .unwrap();

                self.ctx.device.cmd_end_rendering(cb);
            }

            self.ctx.device.cmd_write_timestamp2(
                cb,
                vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
                self.query_pool,
                query_base + PASS_COUNT as u32,
            );

            self.ctx.device.cmd_pipeline_barrier2(
                cb,
                &vk::DependencyInfo::default().image_memory_barriers(&[
                    vk::ImageMemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                        .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                        .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                        .dst_access_mask(vk::AccessFlags2::empty())
                        .old_layout(vk::ImageLayout::ATTACHMENT_OPTIMAL)
                        .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                        .image(self.screen.swapchain_images[self.image_index])
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .level_count(1)
                                .layer_count(1),
                        ),
                ]),
            );
            self.ctx.device.end_command_buffer(cb).unwrap();

            self.ctx
                .device
                .queue_submit(
                    self.ctx.queue,
                    &[vk::SubmitInfo::default()
                        .wait_semaphores(&[self.ctx.image_acquired_semaphores[self.frame_index]])
                        .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
                        .command_buffers(&[cb])
                        .signal_semaphores(&[
                            self.screen.render_complete_semaphores[self.image_index]
                        ])],
                    self.ctx.frame_fences[self.frame_index],
                )
                .unwrap();

            self.frame_index = (self.frame_index + 1) % (MAX_FRAMES_IN_FLIGHT as usize);

            match self.screen.swapchain_loader.queue_present(
                self.ctx.queue,
                &vk::PresentInfoKHR::default()
                    .wait_semaphores(&[self.screen.render_complete_semaphores[self.image_index]])
                    .swapchains(&[self.screen.swapchain])
                    .image_indices(&[self.image_index as u32]),
            ) {
                Ok(suboptimal) => {
                    if suboptimal {
                        self.screen.recreate = true;
                    }
                }
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    self.screen.recreate = true;
                }
                Err(err) => panic!("{err}"),
            }
            self.window.pre_present_notify();

            self.avg_delta_time = self.avg_delta_time.mul_f64(1.0 - FRAME_ACC_ALPHA)
                + delta_time.mul_f64(FRAME_ACC_ALPHA);
            self.frame_id = self.frame_id.wrapping_add(1);

            if self.prev_fixed_time.elapsed().as_secs_f32() >= 1.0 {
                self.prev_fixed_time = time;
                // tick
                // println!("frame time: {:#?}", self.avg_delta_time);
                // println!("FPS: {:.2}", 1.0 / self.avg_delta_time.as_secs_f64());
            }
        };
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            self.ctx.device.device_wait_idle().unwrap();
            // self.device.destroy_image_view(self.depth_view, None);
            // self.allocator
            //     .destroy_image(self.depth_image, &mut self.depth_allocation);
            self.screen
                .swapchain_loader
                .destroy_swapchain(self.screen.swapchain, None);

            self.ctx.device.destroy_device(None);
            self.ctx
                .surface_loader
                .destroy_surface(self.ctx.surface, None);
            self.ctx.instance.destroy_instance(None);
        };
    }
}
