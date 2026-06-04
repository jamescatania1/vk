use std::ffi::{CStr, CString};

use ash::vk::{self};
use glam::{Mat3, Mat4, Quat, Vec3};
use vk_mem::Alloc;
use winit::{
    raw_window_handle::{HasDisplayHandle, HasWindowHandle},
    window::Window,
};

use crate::shaders::Shaders;

pub struct App {
    entry: ash::Entry,
    instance: ash::Instance,

    pub window: Window,
    surface_loader: ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,

    device: ash::Device,
    physical_device: vk::PhysicalDevice,
    queue: vk::Queue,
    queue_family_index: u32,
    allocator: vk_mem::Allocator,

    swapchain_loader: ash::khr::swapchain::Device,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_views: Vec<vk::ImageView>,

    depth_image: vk::Image,
    depth_allocation: vk_mem::Allocation,
    depth_view: vk::ImageView,
    scene: SceneResources,

    pipeline: vk::Pipeline,
    command_buffers: Vec<vk::CommandBuffer>,
    frame_fences: Vec<vk::Fence>,
    image_acquired_semaphores: Vec<vk::Semaphore>,
    frame_complete_semaphores: Vec<vk::Semaphore>,
    image_index: usize,
    frame_index: usize,
}

struct SceneResources {
    vertex_buffers: Vec<(vk::Buffer, vk_mem::Allocation)>,
    index_buffers: Vec<(vk::Buffer, vk_mem::Allocation)>,
    index_counts: Vec<u32>,
    primitive_push_constants: Vec<PushConstants>,
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

#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct PushConstants {
    pub object_id: u32,
    pub material_id: u32,
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct ShaderData {
    view_proj: [[f32; 4]; 4],
}

const MAX_FRAMES_IN_FLIGHT: u32 = 2;

impl SceneResources {
    fn create(allocator: &vk_mem::Allocator) -> Self {
        let (document, buffers, images) = gltf::import("assets/sponza.glb").unwrap();
        let scene = document.default_scene().unwrap();

        let mut vertex_buffers = Vec::new();
        let mut index_buffers = Vec::new();
        let mut index_counts = Vec::new();
        let mut primitive_push_constants = Vec::new();

        for node in scene.nodes() {
            let Some(mesh) = node.mesh() else {
                continue;
            };
            for primitive in mesh.primitives() {
                let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

                let positions = reader.read_positions().unwrap();

                let vertex_count = positions.len();
                let normals = reader.read_normals().unwrap();
                let Some(tangents) = reader.read_tangents() else {
                    eprintln!("no tangents for primitive {}, skipping", primitive.index());
                    continue;
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

                primitive_push_constants.push(PushConstants {
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
            primitive_push_constants,
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
        let _physical_device_properties =
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
            .buffer_device_address(true);
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
        const SWAPCHAIN_FORMAT: vk::Format = vk::Format::B8G8R8A8_SRGB;
        const DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;
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
                        .present_mode(vk::PresentModeKHR::FIFO),
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

        let (depth_image, depth_allocation) = unsafe {
            allocator
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
                .unwrap()
        };
        let depth_view = unsafe {
            device
                .create_image_view(
                    &vk::ImageViewCreateInfo {
                        image: depth_image,
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
                .unwrap()
        };

        let scene = SceneResources::create(&allocator);

        let mut buffers = Vec::new();
        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            let (buffer, allocation) = unsafe {
                allocator
                    .create_buffer(
                        &vk::BufferCreateInfo::default()
                            .size(std::mem::size_of::<ShaderData>() as u64)
                            .usage(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS),
                        &vk_mem::AllocationCreateInfo {
                            flags: vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
                                | vk_mem::AllocationCreateFlags::HOST_ACCESS_ALLOW_TRANSFER_INSTEAD
                                | vk_mem::AllocationCreateFlags::MAPPED,
                            usage: vk_mem::MemoryUsage::Auto,
                            ..Default::default()
                        },
                    )
                    .unwrap()
            };
            let address = unsafe {
                device.get_buffer_device_address(
                    &vk::BufferDeviceAddressInfo::default().buffer(buffer),
                )
            };
            buffers.push((buffer, allocation, address));
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

        let mut frame_complete_semaphores = Vec::new();
        for _ in 0..swapchain_images.len() {
            frame_complete_semaphores
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
                            .size(std::mem::size_of::<vk::DeviceAddress>() as u32)]),
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
                            &vk::PipelineRasterizationStateCreateInfo::default().line_width(1.0),
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

        Self {
            window,
            allocator,
            depth_view,
            entry,
            instance,
            surface_loader,
            surface,
            device,
            physical_device,
            queue,
            queue_family_index,
            swapchain_loader,
            swapchain,
            swapchain_images,
            swapchain_image_views,
            depth_image,
            depth_allocation,
            scene,
            pipeline,
            command_buffers,
            frame_fences,
            image_acquired_semaphores,
            frame_complete_semaphores,
            image_index: 0,
            frame_index: 0,
        }
    }

    pub fn frame(&mut self) {
        unsafe {
            self.device
                .wait_for_fences(&[self.frame_fences[self.frame_index]], true, u64::MAX)
                .unwrap();
            self.device
                .reset_fences(&[self.frame_fences[self.frame_index]])
                .unwrap()
        };

        let (next_image_index, _suboptimal) = unsafe {
            self.swapchain_loader
                .acquire_next_image(
                    self.swapchain,
                    u64::MAX,
                    self.image_acquired_semaphores[self.frame_index],
                    vk::Fence::null(),
                )
                .unwrap()
        };
        self.image_index = next_image_index as usize;

        let size = self.window.inner_size();

        let shader_data = {
            let view =
                glam::Mat4::look_at_rh(glam::vec3(0.0, -1.0, 0.0), glam::Vec3::ZERO, glam::Vec3::Z);
            let mut proj = glam::Mat4::perspective_rh(
                90.0f32.to_radians(),
                size.width as f32 / size.height.max(1) as f32,
                0.1,
                100.0,
            );
            proj.y_axis.y *= -1.0;
            let view_proj = proj * view;
            ShaderData {
                view_proj: view_proj.to_cols_array_2d(),
            }
        };

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

            self.device.cmd_pipeline_barrier2(
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
                        .image(self.depth_image)
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
                                float32: [0.0, 0.0, 1.0, 1.0],
                            },
                        })])
                    .depth_attachment(
                        &vk::RenderingAttachmentInfo::default()
                            .image_view(self.depth_view)
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
            for i in 0..self.scene.vertex_buffers.len() {
                self.device
                    .cmd_bind_vertex_buffers(cb, 0, &[self.scene.vertex_buffers[i].0], &[0]);
                self.device.cmd_bind_index_buffer(
                    cb,
                    self.scene.index_buffers[i].0,
                    0,
                    vk::IndexType::UINT32,
                );
                self.device
                    .cmd_draw_indexed(cb, self.scene.index_counts[i], 1, 0, 0, 0);
            }
            self.device.cmd_end_rendering(cb);
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
                        .signal_semaphores(&[self.frame_complete_semaphores[self.image_index]])],
                    self.frame_fences[self.frame_index],
                )
                .unwrap();

            self.frame_index = (self.frame_index + 1) % (MAX_FRAMES_IN_FLIGHT as usize);

            self.swapchain_loader
                .queue_present(
                    self.queue,
                    &vk::PresentInfoKHR::default()
                        .wait_semaphores(&[self.frame_complete_semaphores[self.image_index]])
                        .swapchains(&[self.swapchain])
                        .image_indices(&[self.image_index as u32]),
                )
                .unwrap();
            self.window.pre_present_notify();
        };
    }
}

impl Drop for App {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();
            self.device.destroy_image_view(self.depth_view, None);
            self.allocator
                .destroy_image(self.depth_image, &mut self.depth_allocation);
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);

            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
        };
    }
}
