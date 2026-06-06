use std::{
    ffi::{CStr, CString},
    time::{Duration, Instant},
};

use ash::vk::{self};
use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Mat4, Quat, Vec3};
use vk_mem::Alloc;
use winit::{
    raw_window_handle::{HasDisplayHandle, HasWindowHandle},
    window::Window,
};

use crate::{camera::Camera, input::Input, shaders::Shaders};

pub struct App {
    entry: ash::Entry,
    instance: ash::Instance,

    pub window: Window,
    surface_loader: ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,

    device: ash::Device,
    physical_device: vk::PhysicalDevice,
    physical_device_properties: vk::PhysicalDeviceProperties,
    queue: vk::Queue,
    allocator: vk_mem::Allocator,

    swapchain_loader: ash::khr::swapchain::Device,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_views: Vec<vk::ImageView>,
    query_pool: vk::QueryPool,

    depth: (vk::Image, vk_mem::Allocation, vk::ImageView),
    color_output: (vk::Image, vk_mem::Allocation, vk::ImageView),
    shadowmap: (vk::Image, vk_mem::Allocation, vk::ImageView, vk::Sampler),
    scene: SceneResources,
    frame_data: Vec<(
        vk::Buffer,
        vk_mem::Allocation,
        *mut FrameData,
        vk::DeviceAddress,
    )>,
    frame_id: u64,
    prev_fixed_time: Instant,
    prev_frame_time: Instant,
    avg_delta_time: Duration,
    avg_gpu_time: Duration,
    pub input: Input,
    camera: Camera,
    sun_azimuth: f64,
    sun_altitude: f64,
    pub cursor_locked: bool,
    pub recreate_swapchain: bool,

    pub imgui: imgui::Context,
    pub imgui_platform: imgui_winit_support::WinitPlatform,
    imgui_renderer: imgui_rs_vulkan_renderer::Renderer,

    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    shadow_pipeline_layout: vk::PipelineLayout,
    shadow_pipeline: vk::Pipeline,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_set: vk::DescriptorSet,
    command_buffers: Vec<vk::CommandBuffer>,
    frame_fences: Vec<vk::Fence>,
    image_acquired_semaphores: Vec<vk::Semaphore>,
    render_complete_semaphores: Vec<vk::Semaphore>,
    image_index: usize,
    frame_index: usize,
}

#[derive(Debug)]
struct SceneResources {
    vertex_buffers: Vec<(vk::Buffer, vk_mem::Allocation)>,
    index_buffers: Vec<(vk::Buffer, vk_mem::Allocation)>,
    index_counts: Vec<u32>,
    images: Vec<(vk::Image, vk_mem::Allocation, vk::ImageView)>,
    samplers: Vec<vk::Sampler>,
    object_buffer: (vk::Buffer, vk_mem::Allocation, vk::DeviceAddress),
    material_buffer: (vk::Buffer, vk_mem::Allocation, vk::DeviceAddress),
    primitive_indices: Vec<PrimitiveIndices>,
}

#[derive(Clone, Copy, Debug)]
struct PrimitiveIndices {
    pub object_id: u32,
    pub material_id: u32,
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct FrameData {
    pub view_proj: [[f32; 4]; 4],
    pub camera_pos: [f32; 3],
    pub sun_view_proj: [[f32; 4]; 4],
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

#[derive(Debug, Clone, Copy, Default, Zeroable, Pod)]
#[repr(C)]
struct PushConstants {
    frame_ptr: vk::DeviceAddress,
    objects_ptr: vk::DeviceAddress,
    materials_ptr: vk::DeviceAddress,
    object_id: u32,
    material_id: u32,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct VertexData {
    position: [f32; 3],
    normal: [f32; 3],
    tangent: [f32; 4],
    color: [f32; 3],
    uv: [f32; 2],
}

const MAX_FRAMES_IN_FLIGHT: u32 = 2;
const QUERY_COUNT: u32 = 2;
const FRAME_ACC_ALPHA: f64 = 1.0 / 30.0;
const ANISOTROPIC_SAMPLES: f32 = 16.0;
const SWAPCHAIN_FORMAT: vk::Format = vk::Format::B8G8R8A8_SRGB;
const DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;
const SHADOWMAP_FORMAT: vk::Format = vk::Format::D32_SFLOAT;
const SHADOWMAP_SIZE: u32 = 2048;

impl SceneResources {
    fn create(
        physical_device_properties: &vk::PhysicalDeviceProperties,
        device: &ash::Device,
        queue: &vk::Queue,
        allocator: &vk_mem::Allocator,
        command_pool: &vk::CommandPool,
    ) -> Self {
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
            let base_transform = Mat4::from_rotation_translation(
                Quat::from_axis_angle(Vec3::X, 90.0f32.to_radians()),
                Vec3::ZERO,
            );
            walk_transform(root, base_transform, &mut transforms);
        }

        let object_buffer = {
            let size = std::mem::size_of::<ObjectData>() * transforms.len();
            let (buffer, allocation) = unsafe {
                allocator
                    .create_buffer(
                        &vk::BufferCreateInfo::default().size(size as u64).usage(
                            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                                | vk::BufferUsageFlags::STORAGE_BUFFER,
                        ),
                        &vk_mem::AllocationCreateInfo {
                            flags: vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
                                | vk_mem::AllocationCreateFlags::MAPPED,
                            usage: vk_mem::MemoryUsage::Auto,
                            ..Default::default()
                        },
                    )
                    .unwrap()
            };
            let mapped = allocator
                .get_allocation_info(&allocation)
                .mapped_data
                .cast::<u8>();

            let transforms = transforms
                .into_iter()
                .map(|transform| ObjectData {
                    transform: transform.to_cols_array_2d(),
                    normal_transform: Mat3::from_mat4(transform)
                        .inverse()
                        .transpose()
                        .to_cols_array_2d(),
                })
                .collect::<Vec<_>>();
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bytemuck::cast_slice(&transforms).as_ptr(),
                    mapped,
                    size,
                );
            }
            allocator
                .flush_allocation(&allocation, 0, size as u64)
                .unwrap();

            let address = unsafe {
                device.get_buffer_device_address(
                    &vk::BufferDeviceAddressInfo::default().buffer(buffer),
                )
            };
            (buffer, allocation, address)
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
                    device
                        .create_sampler(
                            &&&&vk::SamplerCreateInfo::default()
                                .address_mode_u(address_mode_u)
                                .address_mode_v(address_mode_v)
                                .mag_filter(vk::Filter::LINEAR)
                                .min_filter(vk::Filter::LINEAR)
                                .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                                .min_lod(0.0)
                                .max_lod(vk::LOD_CLAMP_NONE)
                                .anisotropy_enable(true)
                                .max_anisotropy(
                                    ANISOTROPIC_SAMPLES.min(
                                        physical_device_properties.limits.max_sampler_anisotropy,
                                    ),
                                ),
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
            let materials = document
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

            let size = std::mem::size_of::<MaterialData>() * materials.len();
            let (buffer, allocation) = unsafe {
                allocator
                    .create_buffer(
                        &vk::BufferCreateInfo::default().size(size as u64).usage(
                            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                                | vk::BufferUsageFlags::STORAGE_BUFFER,
                        ),
                        &vk_mem::AllocationCreateInfo {
                            flags: vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
                                | vk_mem::AllocationCreateFlags::MAPPED,
                            usage: vk_mem::MemoryUsage::Auto,
                            ..Default::default()
                        },
                    )
                    .unwrap()
            };
            let mapped = allocator
                .get_allocation_info(&allocation)
                .mapped_data
                .cast::<u8>();
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bytemuck::cast_slice(&materials).as_ptr(),
                    mapped,
                    size,
                );
            }
            allocator
                .flush_allocation(&allocation, 0, size as u64)
                .unwrap();

            let address = unsafe {
                device.get_buffer_device_address(
                    &vk::BufferDeviceAddressInfo::default().buffer(buffer),
                )
            };
            (buffer, allocation, address)
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
            let (image, allocation) = unsafe {
                allocator
                    .create_image(
                        &vk::ImageCreateInfo::default()
                            .image_type(vk::ImageType::TYPE_2D)
                            .format(format)
                            .extent(vk::Extent3D {
                                width: img.width,
                                height: img.height,
                                depth: 1,
                            })
                            .mip_levels(mip_levels)
                            .array_layers(1)
                            .samples(vk::SampleCountFlags::TYPE_1)
                            .tiling(vk::ImageTiling::OPTIMAL)
                            .usage(
                                vk::ImageUsageFlags::TRANSFER_DST
                                    | vk::ImageUsageFlags::SAMPLED
                                    | vk::ImageUsageFlags::TRANSFER_SRC,
                            )
                            .initial_layout(vk::ImageLayout::UNDEFINED),
                        &vk_mem::AllocationCreateInfo {
                            usage: vk_mem::MemoryUsage::Auto,
                            ..Default::default()
                        },
                    )
                    .unwrap()
            };
            let view = unsafe {
                device
                    .create_image_view(
                        &vk::ImageViewCreateInfo::default()
                            .image(image)
                            .view_type(vk::ImageViewType::TYPE_2D)
                            .format(format)
                            .subresource_range(
                                vk::ImageSubresourceRange::default()
                                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                                    .base_mip_level(0)
                                    .level_count(mip_levels)
                                    .layer_count(1),
                            ),
                        None,
                    )
                    .unwrap()
            };

            let (transfer_buffer, mut transfer_allocation) = unsafe {
                allocator
                    .create_buffer(
                        &vk::BufferCreateInfo::default()
                            .size(pixels.len() as u64)
                            .usage(vk::BufferUsageFlags::TRANSFER_SRC),
                        &vk_mem::AllocationCreateInfo {
                            flags: vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
                                | vk_mem::AllocationCreateFlags::MAPPED,
                            usage: vk_mem::MemoryUsage::Auto,
                            ..Default::default()
                        },
                    )
                    .unwrap()
            };
            let mapped = allocator
                .get_allocation_info(&transfer_allocation)
                .mapped_data
                .cast::<u8>();
            unsafe {
                std::ptr::copy_nonoverlapping(pixels.as_ptr(), mapped, pixels.len());
            }
            allocator
                .flush_allocation(&transfer_allocation, 0, pixels.len() as u64)
                .unwrap();

            let fence = unsafe {
                device
                    .create_fence(&vk::FenceCreateInfo::default(), None)
                    .unwrap()
            };
            let cb = unsafe {
                device
                    .allocate_command_buffers(
                        &vk::CommandBufferAllocateInfo::default()
                            .command_pool(*command_pool)
                            .command_buffer_count(1),
                    )
                    .unwrap()[0]
            };

            unsafe {
                device
                    .begin_command_buffer(
                        cb,
                        &vk::CommandBufferBeginInfo::default()
                            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                    )
                    .unwrap();
                device.cmd_pipeline_barrier2(
                    cb,
                    &vk::DependencyInfo::default().image_memory_barriers(&[
                        vk::ImageMemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::NONE)
                            .src_access_mask(vk::AccessFlags2::NONE)
                            .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                            .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                            .old_layout(vk::ImageLayout::UNDEFINED)
                            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                            .image(image)
                            .subresource_range(
                                vk::ImageSubresourceRange::default()
                                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                                    .base_mip_level(0)
                                    .level_count(mip_levels)
                                    .layer_count(1),
                            ),
                    ]),
                );
                device.cmd_copy_buffer_to_image(
                    cb,
                    transfer_buffer,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[vk::BufferImageCopy::default()
                        .buffer_offset(0)
                        .image_subresource(
                            vk::ImageSubresourceLayers::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .mip_level(0)
                                .layer_count(1),
                        )
                        .image_extent(vk::Extent3D {
                            width: img.width,
                            height: img.height,
                            depth: 1,
                        })],
                );

                // now we generate the mip chain
                let mut w = img.width;
                let mut h = img.height;

                for i in 1..mip_levels {
                    device.cmd_pipeline_barrier2(
                        cb,
                        &vk::DependencyInfo::default().image_memory_barriers(&[
                            vk::ImageMemoryBarrier2::default()
                                .image(image)
                                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                                .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                                .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                                .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                                .subresource_range(
                                    vk::ImageSubresourceRange::default()
                                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                                        .base_mip_level(i - 1)
                                        .level_count(1)
                                        .base_array_layer(0)
                                        .layer_count(1),
                                ),
                        ]),
                    );

                    let dst_w = (w / 2).max(1);
                    let dst_h = (h / 2).max(1);

                    device.cmd_blit_image(
                        cb,
                        image,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &[vk::ImageBlit::default()
                            .src_offsets([
                                vk::Offset3D { x: 0, y: 0, z: 0 },
                                vk::Offset3D {
                                    x: w as i32,
                                    y: h as i32,
                                    z: 1,
                                },
                            ])
                            .src_subresource(
                                vk::ImageSubresourceLayers::default()
                                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                                    .mip_level(i - 1)
                                    .base_array_layer(0)
                                    .layer_count(1),
                            )
                            .dst_offsets([
                                vk::Offset3D { x: 0, y: 0, z: 0 },
                                vk::Offset3D {
                                    x: dst_w as i32,
                                    y: dst_h as i32,
                                    z: 1,
                                },
                            ])
                            .dst_subresource(
                                vk::ImageSubresourceLayers::default()
                                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                                    .mip_level(i)
                                    .base_array_layer(0)
                                    .layer_count(1),
                            )],
                        vk::Filter::LINEAR,
                    );

                    device.cmd_pipeline_barrier2(
                        cb,
                        &vk::DependencyInfo::default().image_memory_barriers(&[
                            vk::ImageMemoryBarrier2::default()
                                .image(image)
                                .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                                .src_access_mask(vk::AccessFlags2::TRANSFER_READ)
                                .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                                .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                                .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                                .subresource_range(
                                    vk::ImageSubresourceRange::default()
                                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                                        .base_mip_level(i - 1)
                                        .level_count(1)
                                        .base_array_layer(0)
                                        .layer_count(1),
                                ),
                        ]),
                    );

                    w = dst_w;
                    h = dst_h;
                }

                device.cmd_pipeline_barrier2(
                    cb,
                    &vk::DependencyInfo::default().image_memory_barriers(&[
                        vk::ImageMemoryBarrier2::default()
                            .image(image)
                            .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                            .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                            .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                            .subresource_range(
                                vk::ImageSubresourceRange::default()
                                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                                    .base_mip_level(mip_levels - 1)
                                    .level_count(1)
                                    .base_array_layer(0)
                                    .layer_count(1),
                            ),
                    ]),
                );

                device.end_command_buffer(cb).unwrap();
                device
                    .queue_submit(
                        *queue,
                        &[vk::SubmitInfo::default().command_buffers(&[cb])],
                        fence,
                    )
                    .unwrap();
                device.wait_for_fences(&[fence], true, u64::MAX).unwrap();
                allocator.destroy_buffer(transfer_buffer, &mut transfer_allocation);
            };

            images.push((image, allocation, view));
        }

        let mut vertex_buffers = Vec::new();
        let mut index_buffers = Vec::new();
        let mut index_counts = Vec::new();
        let mut primitive_indices = Vec::new();

        for node in scene.nodes() {
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

                primitive_indices.push(PrimitiveIndices {
                    object_id: node.index() as u32,
                    material_id: primitive.material().index().unwrap_or(0) as u32,
                });

                let (vertex_buffer, mut vertex_allocation) = unsafe {
                    allocator.create_buffer(
                        &vk::BufferCreateInfo::default()
                            .size((std::mem::size_of::<VertexData>() * vertices.len()) as u64)
                            .usage(vk::BufferUsageFlags::VERTEX_BUFFER),
                        &vk_mem::AllocationCreateInfo {
                            flags: vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
                                | vk_mem::AllocationCreateFlags::HOST_ACCESS_ALLOW_TRANSFER_INSTEAD
                                | vk_mem::AllocationCreateFlags::MAPPED,
                            usage: vk_mem::MemoryUsage::Auto,
                            ..Default::default()
                        },
                    ).unwrap()
                };
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        vertices.as_ptr() as *const u8,
                        allocator.map_memory(&mut vertex_allocation).unwrap(),
                        std::mem::size_of::<VertexData>() * vertices.len(),
                    );
                    allocator.unmap_memory(&mut vertex_allocation);
                }
                vertex_buffers.push((vertex_buffer, vertex_allocation));

                let (index_buffer, mut index_allocation) = unsafe {
                    allocator.create_buffer(
                                        &vk::BufferCreateInfo::default()
                                            .size((std::mem::size_of::<u32>() * indices.len()) as u64)
                                            .usage(vk::BufferUsageFlags::INDEX_BUFFER),
                                        &vk_mem::AllocationCreateInfo {
                                            flags: vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
                                                | vk_mem::AllocationCreateFlags::HOST_ACCESS_ALLOW_TRANSFER_INSTEAD
                                                | vk_mem::AllocationCreateFlags::MAPPED,
                                            usage: vk_mem::MemoryUsage::Auto,
                                            ..Default::default()
                                        },
                                    ).unwrap()
                };
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        indices.as_ptr() as *const u8,
                        allocator.map_memory(&mut index_allocation).unwrap(),
                        std::mem::size_of::<u32>() * indices.len(),
                    );
                    allocator.unmap_memory(&mut index_allocation);
                }
                index_counts.push(indices.len() as u32);
                index_buffers.push((index_buffer, index_allocation));
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
            primitive_indices,
        }
    }
}

impl App {
    pub fn new(window: Window) -> Self {
        let entry = unsafe { ash::Entry::load().unwrap() };

        let app_name = CString::new("vk demo").unwrap();
        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(vk::API_VERSION_1_3);

        let available_extensions =
            unsafe { entry.enumerate_instance_extension_properties(None).unwrap() };
        let has_instance_extension = |name: &CStr| {
            available_extensions.iter().any(|ext| {
                let ext_name = unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) };
                ext_name == name
            })
        };
        let validate =
            cfg!(debug_assertions) && has_instance_extension(&ash::ext::debug_utils::NAME);
        let portability = has_instance_extension(ash::khr::portability_enumeration::NAME);

        let mut extension_names =
            ash_window::enumerate_required_extensions(window.display_handle().unwrap().as_raw())
                .unwrap()
                .to_vec();
        if validate {
            extension_names.push(ash::ext::debug_utils::NAME.as_ptr());
        }
        if portability {
            extension_names.push(ash::khr::portability_enumeration::NAME.as_ptr());
        }

        let validation_layer = CString::new("VK_LAYER_KHRONOS_validation").unwrap();
        let layer_names = if validate {
            vec![validation_layer.as_ptr()]
        } else {
            Vec::new()
        };

        let mut instance_flags = vk::InstanceCreateFlags::empty();
        if portability {
            instance_flags |= vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR;
        }

        let instance = unsafe {
            entry
                .create_instance(
                    &vk::InstanceCreateInfo::default()
                        .application_info(&app_info)
                        .enabled_layer_names(&layer_names)
                        .enabled_extension_names(&extension_names)
                        .flags(instance_flags),
                    None,
                )
                .unwrap()
        };

        let surface = unsafe {
            ash_window::create_surface(
                &entry,
                &instance,
                window.display_handle().unwrap().as_raw(),
                window.window_handle().unwrap().as_raw(),
                None,
            )
            .unwrap()
        };
        let surface_loader = ash::khr::surface::Instance::new(&entry, &instance);

        let physical_device = unsafe { instance.enumerate_physical_devices().unwrap() }
            .into_iter()
            .next()
            .unwrap();
        let physical_device_properties =
            unsafe { instance.get_physical_device_properties(physical_device) };

        let (queue_family_index, _queue_family) =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) }
                .into_iter()
                .enumerate()
                .filter(|(i, q)| {
                    let graphics_valid = q.queue_flags.contains(vk::QueueFlags::GRAPHICS);
                    let surface_valid = unsafe {
                        surface_loader
                            .get_physical_device_surface_support(
                                physical_device,
                                *i as u32,
                                surface,
                            )
                            .unwrap_or(false)
                    };
                    graphics_valid && surface_valid
                })
                .next()
                .map(|(i, q)| (i as u32, q))
                .unwrap();

        let available_device_extensions = unsafe {
            instance
                .enumerate_device_extension_properties(physical_device)
                .unwrap()
        };
        let has_device_extension = |name: &CStr| {
            available_device_extensions.iter().any(|ext| {
                let ext_name = unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) };
                ext_name == name
            })
        };

        let mut device_extensions = Vec::new();
        device_extensions.push(ash::khr::swapchain::NAME.as_ptr());
        if has_device_extension(ash::khr::portability_subset::NAME) {
            device_extensions.push(ash::khr::portability_subset::NAME.as_ptr());
        }

        let vk_10_features = vk::PhysicalDeviceFeatures::default().sampler_anisotropy(true);
        let mut vk_12_features = vk::PhysicalDeviceVulkan12Features::default()
            .descriptor_indexing(true)
            .shader_sampled_image_array_non_uniform_indexing(true)
            .descriptor_binding_variable_descriptor_count(true)
            .runtime_descriptor_array(true)
            .buffer_device_address(true)
            .scalar_block_layout(true);
        let mut vk_13_features = vk::PhysicalDeviceVulkan13Features::default()
            .synchronization2(true)
            .dynamic_rendering(true);

        let p_queue_priorities = [1.0];

        let device = unsafe {
            instance
                .create_device(
                    physical_device,
                    &vk::DeviceCreateInfo::default()
                        .queue_create_infos(&[vk::DeviceQueueCreateInfo {
                            queue_family_index,
                            queue_count: 1,
                            p_queue_priorities: p_queue_priorities.as_ptr(),
                            ..Default::default()
                        }])
                        .enabled_extension_names(&device_extensions)
                        .enabled_features(&vk_10_features)
                        .push_next(&mut vk_13_features)
                        .push_next(&mut vk_12_features),
                    None,
                )
                .unwrap()
        };
        let queue = unsafe {
            device.get_device_queue2(
                &vk::DeviceQueueInfo2::default()
                    .queue_family_index(queue_family_index)
                    .queue_index(0),
            )
        };

        let allocator = unsafe {
            let mut allocator_create_info =
                vk_mem::AllocatorCreateInfo::new(&instance, &device, physical_device);
            allocator_create_info.flags = vk_mem::AllocatorCreateFlags::BUFFER_DEVICE_ADDRESS;
            vk_mem::Allocator::new(allocator_create_info)
        }
        .unwrap();

        let swapchain_loader = ash::khr::swapchain::Device::new(&instance, &device);
        let surface_capabilities = unsafe {
            surface_loader
                .get_physical_device_surface_capabilities(physical_device, surface)
                .unwrap()
        };

        let (swapchain, swapchain_images, swapchain_image_views) = unsafe {
            let swapchain = swapchain_loader
                .create_swapchain(
                    &vk::SwapchainCreateInfoKHR::default()
                        .surface(surface)
                        .min_image_count(surface_capabilities.min_image_count)
                        .image_format(SWAPCHAIN_FORMAT)
                        .image_color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR)
                        .image_extent(surface_capabilities.current_extent)
                        .image_array_layers(1)
                        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                        .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
                        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                        .present_mode(vk::PresentModeKHR::IMMEDIATE),
                    None,
                )
                .unwrap();
            let images = swapchain_loader.get_swapchain_images(swapchain).unwrap();
            let views = images
                .iter()
                .map(|img| {
                    device
                        .create_image_view(
                            &vk::ImageViewCreateInfo::default()
                                .image(*img)
                                .view_type(vk::ImageViewType::TYPE_2D)
                                .format(SWAPCHAIN_FORMAT)
                                .subresource_range(
                                    vk::ImageSubresourceRange::default()
                                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                                        .level_count(1)
                                        .layer_count(1),
                                ),
                            None,
                        )
                        .unwrap()
                })
                .collect::<Vec<_>>();

            (swapchain, images, views)
        };

        let depth = unsafe {
            let (image, allocation) = allocator
                .create_image(
                    &vk::ImageCreateInfo::default()
                        .image_type(vk::ImageType::TYPE_2D)
                        .format(DEPTH_FORMAT)
                        .extent(surface_capabilities.current_extent.into())
                        .mip_levels(1)
                        .array_layers(1)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .tiling(vk::ImageTiling::OPTIMAL)
                        .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
                        .initial_layout(vk::ImageLayout::UNDEFINED),
                    &vk_mem::AllocationCreateInfo {
                        flags: vk_mem::AllocationCreateFlags::DEDICATED_MEMORY,
                        usage: vk_mem::MemoryUsage::Auto,
                        ..Default::default()
                    },
                )
                .unwrap();
            let view = device
                .create_image_view(
                    &vk::ImageViewCreateInfo {
                        image,
                        view_type: vk::ImageViewType::TYPE_2D,
                        format: DEPTH_FORMAT,
                        subresource_range: vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::DEPTH)
                            .level_count(1)
                            .layer_count(1),
                        ..Default::default()
                    },
                    None,
                )
                .unwrap();
            (image, allocation, view)
        };

        let color_output = unsafe {
            let (image, allocation) = allocator
                .create_image(
                    &vk::ImageCreateInfo::default()
                        .image_type(vk::ImageType::TYPE_2D)
                        .format(vk::Format::R16G16B16A16_SFLOAT)
                        .extent(surface_capabilities.current_extent.into())
                        .mip_levels(1)
                        .array_layers(1)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .tiling(vk::ImageTiling::OPTIMAL)
                        .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED)
                        .initial_layout(vk::ImageLayout::UNDEFINED),
                    &vk_mem::AllocationCreateInfo {
                        flags: vk_mem::AllocationCreateFlags::DEDICATED_MEMORY,
                        usage: vk_mem::MemoryUsage::Auto,
                        ..Default::default()
                    },
                )
                .unwrap();
            let view = device
                .create_image_view(
                    &vk::ImageViewCreateInfo {
                        image,
                        view_type: vk::ImageViewType::TYPE_2D,
                        format: vk::Format::R16G16B16A16_SFLOAT,
                        subresource_range: vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .level_count(1)
                            .layer_count(1),
                        ..Default::default()
                    },
                    None,
                )
                .unwrap();
            (image, allocation, view)
        };

        let shadowmap = unsafe {
            let (image, allocation) = allocator
                .create_image(
                    &vk::ImageCreateInfo::default()
                        .image_type(vk::ImageType::TYPE_2D)
                        .format(SHADOWMAP_FORMAT)
                        .extent(vk::Extent3D {
                            width: SHADOWMAP_SIZE,
                            height: SHADOWMAP_SIZE,
                            depth: 1,
                        })
                        .mip_levels(1)
                        .array_layers(1)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .tiling(vk::ImageTiling::OPTIMAL)
                        .usage(
                            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                                | vk::ImageUsageFlags::SAMPLED,
                        )
                        .initial_layout(vk::ImageLayout::UNDEFINED),
                    &vk_mem::AllocationCreateInfo {
                        flags: vk_mem::AllocationCreateFlags::DEDICATED_MEMORY,
                        usage: vk_mem::MemoryUsage::Auto,
                        ..Default::default()
                    },
                )
                .unwrap();
            let view = device
                .create_image_view(
                    &vk::ImageViewCreateInfo {
                        image,
                        view_type: vk::ImageViewType::TYPE_2D,
                        format: SHADOWMAP_FORMAT,
                        subresource_range: vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::DEPTH)
                            .level_count(1)
                            .layer_count(1),
                        ..Default::default()
                    },
                    None,
                )
                .unwrap();
            let sampler = device
                .create_sampler(
                    &vk::SamplerCreateInfo::default()
                        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_BORDER)
                        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_BORDER)
                        .border_color(vk::BorderColor::FLOAT_OPAQUE_WHITE)
                        .anisotropy_enable(false)
                        .mipmap_mode(vk::SamplerMipmapMode::NEAREST),
                    None,
                )
                .unwrap();
            (image, allocation, view, sampler)
        };

        let mut render_complete_semaphores = Vec::new();
        for _ in 0..swapchain_images.len() {
            render_complete_semaphores
                .push(unsafe { device.create_semaphore(&Default::default(), None).unwrap() });
        }

        let command_pool = unsafe {
            device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo {
                        flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                        queue_family_index,
                        ..Default::default()
                    },
                    None,
                )
                .unwrap()
        };
        let command_buffers = unsafe {
            device
                .allocate_command_buffers(&vk::CommandBufferAllocateInfo {
                    command_pool,
                    command_buffer_count: MAX_FRAMES_IN_FLIGHT,
                    ..Default::default()
                })
                .unwrap()
        };

        let query_pool = unsafe {
            device
                .create_query_pool(
                    &vk::QueryPoolCreateInfo::default()
                        .query_type(vk::QueryType::TIMESTAMP)
                        .query_count(MAX_FRAMES_IN_FLIGHT * QUERY_COUNT),
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
            &instance,
            physical_device,
            device.clone(),
            queue,
            command_pool,
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

        let scene = SceneResources::create(
            &physical_device_properties,
            &device,
            &queue,
            &allocator,
            &command_pool,
        );

        let descriptor_pool = unsafe {
            device
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
                                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                                .descriptor_count(1),
                        ]),
                    None,
                )
                .unwrap()
        };
        let descriptor_set_layout = unsafe {
            device
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
                            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                            .descriptor_count(1)
                            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
                    ]),
                    None,
                )
                .unwrap()
        };
        let descriptor_set = unsafe {
            device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(descriptor_pool)
                        .set_layouts(&[descriptor_set_layout]),
                )
                .unwrap()[0]
        };
        let image_infos = scene
            .images
            .iter()
            .map(|(_, _, view)| {
                vk::DescriptorImageInfo::default()
                    .image_view(*view)
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            })
            .collect::<Vec<_>>();
        let sampler_infos = scene
            .samplers
            .iter()
            .map(|sampler| vk::DescriptorImageInfo::default().sampler(*sampler))
            .collect::<Vec<_>>();
        let shadowmap_info = vk::DescriptorImageInfo::default()
            .image_view(shadowmap.2)
            .image_layout(vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL)
            .sampler(shadowmap.3);
        unsafe {
            device.update_descriptor_sets(
                &[
                    vk::WriteDescriptorSet::default()
                        .dst_set(descriptor_set)
                        .dst_binding(0)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                        .image_info(&image_infos),
                    vk::WriteDescriptorSet::default()
                        .dst_set(descriptor_set)
                        .dst_binding(1)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::SAMPLER)
                        .image_info(&sampler_infos),
                    vk::WriteDescriptorSet::default()
                        .dst_set(descriptor_set)
                        .dst_binding(2)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(&[shadowmap_info]),
                ],
                &[],
            )
        };

        let mut frame_data = Vec::new();
        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            let (buffer, allocation) = unsafe {
                allocator
                    .create_buffer(
                        &vk::BufferCreateInfo::default()
                            .size(std::mem::size_of::<FrameData>() as u64)
                            .usage(
                                vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                                    | vk::BufferUsageFlags::STORAGE_BUFFER,
                            ),
                        &vk_mem::AllocationCreateInfo {
                            flags: vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
                                | vk_mem::AllocationCreateFlags::MAPPED,
                            usage: vk_mem::MemoryUsage::Auto,
                            ..Default::default()
                        },
                    )
                    .unwrap()
            };
            let mapped = allocator
                .get_allocation_info(&allocation)
                .mapped_data
                .cast::<FrameData>();
            let address = unsafe {
                device.get_buffer_device_address(
                    &vk::BufferDeviceAddressInfo::default().buffer(buffer),
                )
            };
            frame_data.push((buffer, allocation, mapped, address));
        }

        let mut frame_fences = Vec::new();
        let mut image_acquired_semaphores = Vec::new();

        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            frame_fences.push(unsafe {
                device
                    .create_fence(
                        &vk::FenceCreateInfo {
                            flags: vk::FenceCreateFlags::SIGNALED,
                            ..Default::default()
                        },
                        None,
                    )
                    .unwrap()
            });
            image_acquired_semaphores
                .push(unsafe { device.create_semaphore(&Default::default(), None).unwrap() });
        }

        let shaders = Shaders::new();
        let shader_vertex = unsafe {
            device
                .create_shader_module(
                    &vk::ShaderModuleCreateInfo::default().code(&shaders.main.vertex),
                    None,
                )
                .unwrap()
        };
        let shader_fragment = unsafe {
            device
                .create_shader_module(
                    &vk::ShaderModuleCreateInfo::default().code(&shaders.main.fragment),
                    None,
                )
                .unwrap()
        };

        let pipeline_layout = unsafe {
            device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .push_constant_ranges(&[vk::PushConstantRange::default()
                            .stage_flags(
                                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                            )
                            .size(std::mem::size_of::<PushConstants>() as u32)])
                        .set_layouts(&[descriptor_set_layout]),
                    None,
                )
                .unwrap()
        };

        let pipeline = unsafe {
            device
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::GraphicsPipelineCreateInfo::default()
                        .push_next(
                            &mut vk::PipelineRenderingCreateInfo::default()
                                .color_attachment_formats(&[SWAPCHAIN_FORMAT])
                                .depth_attachment_format(DEPTH_FORMAT),
                        )
                        .stages(&[
                            vk::PipelineShaderStageCreateInfo::default()
                                .stage(vk::ShaderStageFlags::VERTEX)
                                .name(&CString::new("main").unwrap())
                                .module(shader_vertex),
                            vk::PipelineShaderStageCreateInfo::default()
                                .stage(vk::ShaderStageFlags::FRAGMENT)
                                .name(&CString::new("main").unwrap())
                                .module(shader_fragment),
                        ])
                        .vertex_input_state(
                            &vk::PipelineVertexInputStateCreateInfo::default()
                                .vertex_binding_descriptions(&[
                                    vk::VertexInputBindingDescription::default()
                                        .binding(0)
                                        .stride(std::mem::size_of::<VertexData>() as u32)
                                        .input_rate(vk::VertexInputRate::VERTEX),
                                ])
                                .vertex_attribute_descriptions(&[
                                    vk::VertexInputAttributeDescription::default()
                                        .location(0)
                                        .binding(0)
                                        .format(vk::Format::R32G32B32_SFLOAT)
                                        .offset(std::mem::offset_of!(VertexData, position) as u32),
                                    vk::VertexInputAttributeDescription::default()
                                        .location(1)
                                        .binding(0)
                                        .format(vk::Format::R32G32B32_SFLOAT)
                                        .offset(std::mem::offset_of!(VertexData, normal) as u32),
                                    vk::VertexInputAttributeDescription::default()
                                        .location(2)
                                        .binding(0)
                                        .format(vk::Format::R32G32B32A32_SFLOAT)
                                        .offset(std::mem::offset_of!(VertexData, tangent) as u32),
                                    vk::VertexInputAttributeDescription::default()
                                        .location(3)
                                        .binding(0)
                                        .format(vk::Format::R32G32B32_SFLOAT)
                                        .offset(std::mem::offset_of!(VertexData, color) as u32),
                                    vk::VertexInputAttributeDescription::default()
                                        .location(4)
                                        .binding(0)
                                        .format(vk::Format::R32G32_SFLOAT)
                                        .offset(std::mem::offset_of!(VertexData, uv) as u32),
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

        let shader_shadow_vertex = unsafe {
            device
                .create_shader_module(
                    &vk::ShaderModuleCreateInfo::default().code(&shaders.shadow.vertex),
                    None,
                )
                .unwrap()
        };
        let shader_shadow_fragment = unsafe {
            device
                .create_shader_module(
                    &vk::ShaderModuleCreateInfo::default().code(&shaders.shadow.fragment),
                    None,
                )
                .unwrap()
        };

        let shadow_pipeline_layout = unsafe {
            device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&[
                        vk::PushConstantRange::default()
                            .stage_flags(vk::ShaderStageFlags::VERTEX)
                            .size(std::mem::size_of::<PushConstants>() as u32),
                    ]),
                    None,
                )
                .unwrap()
        };

        let shadow_pipeline = unsafe {
            device
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
                                .module(shader_shadow_vertex),
                            vk::PipelineShaderStageCreateInfo::default()
                                .stage(vk::ShaderStageFlags::FRAGMENT)
                                .name(&CString::new("main").unwrap())
                                .module(shader_shadow_fragment),
                        ])
                        .vertex_input_state(
                            &vk::PipelineVertexInputStateCreateInfo::default()
                                .vertex_binding_descriptions(&[
                                    vk::VertexInputBindingDescription::default()
                                        .binding(0)
                                        .stride(std::mem::size_of::<VertexData>() as u32)
                                        .input_rate(vk::VertexInputRate::VERTEX),
                                ])
                                .vertex_attribute_descriptions(&[
                                    vk::VertexInputAttributeDescription::default()
                                        .location(0)
                                        .binding(0)
                                        .format(vk::Format::R32G32B32_SFLOAT)
                                        .offset(std::mem::offset_of!(VertexData, position) as u32),
                                    vk::VertexInputAttributeDescription::default()
                                        .location(1)
                                        .binding(0)
                                        .format(vk::Format::R32G32B32_SFLOAT)
                                        .offset(std::mem::offset_of!(VertexData, normal) as u32),
                                    vk::VertexInputAttributeDescription::default()
                                        .location(2)
                                        .binding(0)
                                        .format(vk::Format::R32G32B32A32_SFLOAT)
                                        .offset(std::mem::offset_of!(VertexData, tangent) as u32),
                                    vk::VertexInputAttributeDescription::default()
                                        .location(3)
                                        .binding(0)
                                        .format(vk::Format::R32G32B32_SFLOAT)
                                        .offset(std::mem::offset_of!(VertexData, color) as u32),
                                    vk::VertexInputAttributeDescription::default()
                                        .location(4)
                                        .binding(0)
                                        .format(vk::Format::R32G32_SFLOAT)
                                        .offset(std::mem::offset_of!(VertexData, uv) as u32),
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

        Self {
            entry,
            window,
            allocator,
            instance,
            surface_loader,
            surface,
            device,
            physical_device,
            physical_device_properties,
            queue,
            swapchain_loader,
            swapchain,
            swapchain_images,
            swapchain_image_views,
            depth,
            color_output,
            shadowmap,
            frame_data,
            scene,
            frame_id: 0,
            prev_frame_time: Instant::now(),
            prev_fixed_time: Instant::now(),
            avg_delta_time: Duration::ZERO,
            avg_gpu_time: Duration::ZERO,
            input: Default::default(),
            camera: Camera::new(),
            sun_azimuth: 45.0,
            sun_altitude: 45.0,
            cursor_locked: false,
            recreate_swapchain: false,
            pipeline_layout,
            pipeline,
            shadow_pipeline_layout,
            shadow_pipeline,
            imgui,
            imgui_platform,
            imgui_renderer,
            query_pool,
            descriptor_pool,
            descriptor_set_layout,
            descriptor_set,
            command_buffers,
            frame_fences,
            image_acquired_semaphores,
            render_complete_semaphores,
            image_index: 0,
            frame_index: 0,
        }
    }

    pub fn frame(&mut self) {
        let time = Instant::now();
        let delta_time = time - self.prev_frame_time;
        self.prev_frame_time = time;

        unsafe {
            self.device
                .wait_for_fences(&[self.frame_fences[self.frame_index]], true, u64::MAX)
                .unwrap();
        };

        let query_base = self.frame_index as u32 * QUERY_COUNT;

        if self.frame_id >= MAX_FRAMES_IN_FLIGHT as u64 {
            let mut timestamps = [0u64; QUERY_COUNT as usize];
            unsafe {
                match self.device.get_query_pool_results(
                    self.query_pool,
                    query_base,
                    &mut timestamps,
                    vk::QueryResultFlags::TYPE_64,
                ) {
                    Ok(()) => {
                        let delta = Duration::from_nanos(
                            (timestamps[1].wrapping_sub(timestamps[0]) as f64
                                * self.physical_device_properties.limits.timestamp_period as f64)
                                as u64,
                        );
                        self.avg_gpu_time = self.avg_gpu_time.mul_f64(1.0 - FRAME_ACC_ALPHA)
                            + delta.mul_f64(FRAME_ACC_ALPHA);
                    }
                    Err(vk::Result::NOT_READY) => {}
                    Err(err) => panic!("{err:?}"),
                }
            };
        }

        if self.recreate_swapchain {
            self.recreate_swapchain();
        }

        let (next_image_index, _suboptimal) = unsafe {
            match self.swapchain_loader.acquire_next_image(
                self.swapchain,
                u64::MAX,
                self.image_acquired_semaphores[self.frame_index],
                vk::Fence::null(),
            ) {
                Ok(res) => res,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    self.recreate_swapchain = true;
                    return;
                }
                Err(err) => {
                    panic!("{err}");
                }
            }
        };

        unsafe {
            self.device
                .reset_fences(&[self.frame_fences[self.frame_index]])
                .unwrap()
        }
        self.image_index = next_image_index as usize;

        let size = self.window.inner_size();

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

        self.camera.update(
            glam::uvec2(size.width, size.height),
            &delta_time,
            &self.input,
            self.cursor_locked,
        );

        self.input.update();

        unsafe {
            let mapped = self.frame_data[self.frame_index].2;
            (*mapped).view_proj = self.camera.view_proj.to_cols_array_2d();
            (*mapped).camera_pos = self.camera.position.as_vec3().to_array();

            let az = self.sun_azimuth.to_radians() as f32;
            let alt = self.sun_altitude.to_radians() as f32;
            let sun_dir =
                glam::vec3(alt.cos() * az.sin(), alt.cos() * az.cos(), alt.sin()).normalize();

            const SHADOW_RADIUS: f32 = 20.0;
            let center =
                self.camera.position.as_vec3() + self.camera.forward.as_vec3() * SHADOW_RADIUS;
            let light_pos = center + sun_dir * 200.0;

            let sun_view = glam::Mat4::look_at_rh(light_pos, center, glam::Vec3::Z);

            let sun_proj = glam::Mat4::orthographic_rh(
                -SHADOW_RADIUS,
                SHADOW_RADIUS,
                -SHADOW_RADIUS,
                SHADOW_RADIUS,
                0.0,
                400.0,
            );
            let sun_view_proj = sun_proj * sun_view;
            (*mapped).sun_view_proj = sun_view_proj.to_cols_array_2d();
        }
        self.allocator
            .flush_allocation(
                &self.frame_data[self.frame_index].1,
                0,
                std::mem::size_of::<FrameData>() as u64,
            )
            .unwrap();

        let cb = self.command_buffers[self.frame_index];

        unsafe {
            self.device
                .reset_command_buffer(cb, vk::CommandBufferResetFlags::empty())
                .unwrap();
            self.device
                .begin_command_buffer(
                    cb,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .unwrap();

            self.device
                .cmd_reset_query_pool(cb, self.query_pool, query_base, QUERY_COUNT);
            self.device.cmd_write_timestamp2(
                cb,
                vk::PipelineStageFlags2::TOP_OF_PIPE,
                self.query_pool,
                query_base,
            );

            self.device.cmd_pipeline_barrier2(
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
                        .image(self.shadowmap.0)
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(vk::ImageAspectFlags::DEPTH)
                                .level_count(1)
                                .layer_count(1),
                        ),
                ]),
            );

            self.device.cmd_begin_rendering(
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
                            .image_view(self.shadowmap.2)
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
            self.device.cmd_set_viewport(
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
            self.device.cmd_set_scissor(
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
            self.device.cmd_bind_pipeline(
                cb,
                vk::PipelineBindPoint::GRAPHICS,
                self.shadow_pipeline,
            );
            for i in 0..self.scene.vertex_buffers.len() {
                self.device
                    .cmd_bind_vertex_buffers(cb, 0, &[self.scene.vertex_buffers[i].0], &[0]);
                self.device.cmd_bind_index_buffer(
                    cb,
                    self.scene.index_buffers[i].0,
                    0,
                    vk::IndexType::UINT32,
                );
                self.device.cmd_push_constants(
                    cb,
                    self.shadow_pipeline_layout,
                    vk::ShaderStageFlags::VERTEX,
                    0,
                    bytemuck::bytes_of(&PushConstants {
                        frame_ptr: self.frame_data[self.frame_index].3,
                        objects_ptr: self.scene.object_buffer.2,
                        materials_ptr: self.scene.material_buffer.2,
                        object_id: self.scene.primitive_indices[i].object_id,
                        material_id: self.scene.primitive_indices[i].material_id,
                    }),
                );
                self.device
                    .cmd_draw_indexed(cb, self.scene.index_counts[i], 1, 0, 0, 0);
            }

            self.device.cmd_end_rendering(cb);

            self.device.cmd_pipeline_barrier2(
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
                        .image(self.shadowmap.0)
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(vk::ImageAspectFlags::DEPTH)
                                .level_count(1)
                                .layer_count(1),
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
                        .image(self.swapchain_images[self.image_index])
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
                        .image(self.depth.0)
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(vk::ImageAspectFlags::DEPTH)
                                .level_count(1)
                                .layer_count(1),
                        ),
                ]),
            );
            self.device.cmd_begin_rendering(
                cb,
                &vk::RenderingInfo::default()
                    .render_area(vk::Rect2D::default().extent(vk::Extent2D {
                        width: size.width,
                        height: size.height,
                    }))
                    .layer_count(1)
                    .color_attachments(&[vk::RenderingAttachmentInfo::default()
                        .image_view(self.swapchain_image_views[self.image_index])
                        .image_layout(vk::ImageLayout::ATTACHMENT_OPTIMAL)
                        .load_op(vk::AttachmentLoadOp::CLEAR)
                        .store_op(vk::AttachmentStoreOp::STORE)
                        .clear_value(vk::ClearValue {
                            color: vk::ClearColorValue {
                                float32: [0.0, 0.0, 0.0, 1.0],
                            },
                        })])
                    .depth_attachment(
                        &vk::RenderingAttachmentInfo::default()
                            .image_view(self.depth.2)
                            .image_layout(vk::ImageLayout::ATTACHMENT_OPTIMAL)
                            .load_op(vk::AttachmentLoadOp::CLEAR)
                            .store_op(vk::AttachmentStoreOp::DONT_CARE)
                            .clear_value(vk::ClearValue {
                                depth_stencil: vk::ClearDepthStencilValue {
                                    depth: 1.0,
                                    stencil: 0,
                                },
                            }),
                    ),
            );
            self.device.cmd_set_viewport(
                cb,
                0,
                &[vk::Viewport {
                    width: size.width as f32,
                    height: size.height as f32,
                    x: 0.0,
                    y: 0.0,
                    min_depth: 0.0,
                    max_depth: 1.0,
                }],
            );
            self.device.cmd_set_scissor(
                cb,
                0,
                &[vk::Rect2D {
                    extent: vk::Extent2D {
                        width: size.width,
                        height: size.height,
                    },
                    offset: vk::Offset2D { x: 0, y: 0 },
                }],
            );
            self.device
                .cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            self.device.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[self.descriptor_set],
                &[],
            );
            for i in 0..self.scene.vertex_buffers.len() {
                self.device
                    .cmd_bind_vertex_buffers(cb, 0, &[self.scene.vertex_buffers[i].0], &[0]);
                self.device.cmd_bind_index_buffer(
                    cb,
                    self.scene.index_buffers[i].0,
                    0,
                    vk::IndexType::UINT32,
                );
                self.device.cmd_push_constants(
                    cb,
                    self.pipeline_layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    0,
                    bytemuck::bytes_of(&PushConstants {
                        frame_ptr: self.frame_data[self.frame_index].3,
                        objects_ptr: self.scene.object_buffer.2,
                        materials_ptr: self.scene.material_buffer.2,
                        object_id: self.scene.primitive_indices[i].object_id,
                        material_id: self.scene.primitive_indices[i].material_id,
                    }),
                );
                self.device
                    .cmd_draw_indexed(cb, self.scene.index_counts[i], 1, 0, 0, 0);
            }

            self.imgui_platform
                .prepare_frame(self.imgui.io_mut(), &self.window)
                .unwrap();
            let ui = self.imgui.frame();
            ui.window("Debug")
                .size([300.0, 200.0], imgui::Condition::FirstUseEver)
                .position([0.0, 0.0], imgui::Condition::FirstUseEver)
                .build(|| {
                    ui.text(format!("Resolution: {} x {}", size.width, size.height));
                    ui.text(format!(
                        "FPS: {:.1}",
                        1.0 / self.avg_delta_time.as_secs_f64()
                    ));
                    ui.separator();
                    ui.text(format!("Frame Time: {:#?}", self.avg_delta_time,));
                    ui.text(format!("GPU Time: {:#?}", self.avg_gpu_time,));
                    ui.separator();
                    ui.slider("Sun Azimuth", 0.0, 360.0, &mut self.sun_azimuth);
                    ui.slider("Sun Altitude", 0.0, 90.0, &mut self.sun_altitude);
                });
            self.imgui_platform.prepare_render(&ui, &self.window);
            self.imgui_renderer
                .cmd_draw(cb, imgui::Context::render(&mut self.imgui))
                .unwrap();

            self.device.cmd_end_rendering(cb);

            self.device.cmd_write_timestamp2(
                cb,
                vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
                self.query_pool,
                query_base + 1,
            );

            self.device.cmd_pipeline_barrier2(
                cb,
                &vk::DependencyInfo::default().image_memory_barriers(&[
                    vk::ImageMemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                        .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                        .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                        .dst_access_mask(vk::AccessFlags2::empty())
                        .old_layout(vk::ImageLayout::ATTACHMENT_OPTIMAL)
                        .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                        .image(self.swapchain_images[self.image_index])
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .level_count(1)
                                .layer_count(1),
                        ),
                ]),
            );
            self.device.end_command_buffer(cb).unwrap();

            self.device
                .queue_submit(
                    self.queue,
                    &[vk::SubmitInfo::default()
                        .wait_semaphores(&[self.image_acquired_semaphores[self.frame_index]])
                        .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
                        .command_buffers(&[cb])
                        .signal_semaphores(&[self.render_complete_semaphores[self.image_index]])],
                    self.frame_fences[self.frame_index],
                )
                .unwrap();

            self.frame_index = (self.frame_index + 1) % (MAX_FRAMES_IN_FLIGHT as usize);

            match self.swapchain_loader.queue_present(
                self.queue,
                &vk::PresentInfoKHR::default()
                    .wait_semaphores(&[self.render_complete_semaphores[self.image_index]])
                    .swapchains(&[self.swapchain])
                    .image_indices(&[self.image_index as u32]),
            ) {
                Ok(suboptimal) => {
                    if suboptimal {
                        self.recreate_swapchain = true;
                    }
                }
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    self.recreate_swapchain = true;
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

    fn recreate_swapchain(&mut self) {
        let size = self.window.inner_size();
        if size.width == 0 || size.height == 0 {
            return;
        }

        self.recreate_swapchain = false;
        unsafe { self.device.device_wait_idle().unwrap() };

        let old_swapchain = self.swapchain;

        for sem in self.render_complete_semaphores.drain(..) {
            unsafe { self.device.destroy_semaphore(sem, None) };
        }
        for view in self.swapchain_image_views.drain(..) {
            unsafe { self.device.destroy_image_view(view, None) };
        }
        unsafe {
            self.device.destroy_image_view(self.depth.2, None);
            self.allocator
                .destroy_image(self.depth.0, &mut self.depth.1);

            self.device.destroy_image_view(self.color_output.2, None);
            self.allocator
                .destroy_image(self.color_output.0, &mut self.color_output.1);
        }

        let surface_capabilities = unsafe {
            self.surface_loader
                .get_physical_device_surface_capabilities(self.physical_device, self.surface)
                .unwrap()
        };

        self.swapchain = unsafe {
            self.swapchain_loader
                .create_swapchain(
                    &vk::SwapchainCreateInfoKHR::default()
                        .surface(self.surface)
                        .min_image_count(surface_capabilities.min_image_count)
                        .image_format(SWAPCHAIN_FORMAT)
                        .image_color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR)
                        .image_extent(surface_capabilities.current_extent)
                        .image_array_layers(1)
                        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                        .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
                        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                        .present_mode(vk::PresentModeKHR::IMMEDIATE)
                        .old_swapchain(old_swapchain),
                    None,
                )
                .unwrap()
        };

        unsafe { self.swapchain_loader.destroy_swapchain(old_swapchain, None) };

        self.swapchain_images = unsafe {
            self.swapchain_loader
                .get_swapchain_images(self.swapchain)
                .unwrap()
        };
        self.swapchain_image_views = self
            .swapchain_images
            .iter()
            .map(|img| unsafe {
                self.device
                    .create_image_view(
                        &vk::ImageViewCreateInfo::default()
                            .image(*img)
                            .view_type(vk::ImageViewType::TYPE_2D)
                            .format(SWAPCHAIN_FORMAT)
                            .subresource_range(
                                vk::ImageSubresourceRange::default()
                                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                                    .level_count(1)
                                    .layer_count(1),
                            ),
                        None,
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();

        self.depth = unsafe {
            let (image, allocation) = self
                .allocator
                .create_image(
                    &vk::ImageCreateInfo::default()
                        .image_type(vk::ImageType::TYPE_2D)
                        .format(DEPTH_FORMAT)
                        .extent(surface_capabilities.current_extent.into())
                        .mip_levels(1)
                        .array_layers(1)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .tiling(vk::ImageTiling::OPTIMAL)
                        .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
                        .initial_layout(vk::ImageLayout::UNDEFINED),
                    &vk_mem::AllocationCreateInfo {
                        flags: vk_mem::AllocationCreateFlags::DEDICATED_MEMORY,
                        usage: vk_mem::MemoryUsage::Auto,
                        ..Default::default()
                    },
                )
                .unwrap();
            let view = self
                .device
                .create_image_view(
                    &vk::ImageViewCreateInfo {
                        image: image,
                        view_type: vk::ImageViewType::TYPE_2D,
                        format: DEPTH_FORMAT,
                        subresource_range: vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::DEPTH)
                            .level_count(1)
                            .layer_count(1),
                        ..Default::default()
                    },
                    None,
                )
                .unwrap();
            (image, allocation, view)
        };

        self.color_output = unsafe {
            let (image, allocation) = self
                .allocator
                .create_image(
                    &vk::ImageCreateInfo::default()
                        .image_type(vk::ImageType::TYPE_2D)
                        .format(vk::Format::R16G16B16A16_SFLOAT)
                        .extent(surface_capabilities.current_extent.into())
                        .mip_levels(1)
                        .array_layers(1)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .tiling(vk::ImageTiling::OPTIMAL)
                        .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED)
                        .initial_layout(vk::ImageLayout::UNDEFINED),
                    &vk_mem::AllocationCreateInfo {
                        flags: vk_mem::AllocationCreateFlags::DEDICATED_MEMORY,
                        usage: vk_mem::MemoryUsage::Auto,
                        ..Default::default()
                    },
                )
                .unwrap();
            let view = self
                .device
                .create_image_view(
                    &vk::ImageViewCreateInfo {
                        image,
                        view_type: vk::ImageViewType::TYPE_2D,
                        format: vk::Format::R16G16B16A16_SFLOAT,
                        subresource_range: vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .level_count(1)
                            .layer_count(1),
                        ..Default::default()
                    },
                    None,
                )
                .unwrap();
            (image, allocation, view)
        };

        self.render_complete_semaphores = Vec::new();
        for _ in 0..self.swapchain_images.len() {
            self.render_complete_semaphores.push(unsafe {
                self.device
                    .create_semaphore(&Default::default(), None)
                    .unwrap()
            });
        }
    }
}

impl Drop for App {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();
            // self.device.destroy_image_view(self.depth_view, None);
            // self.allocator
            //     .destroy_image(self.depth_image, &mut self.depth_allocation);
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);

            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
        };
    }
}
