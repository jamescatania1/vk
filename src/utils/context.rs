use std::ffi::{CStr, CString};

use ash::vk;
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};

pub trait VkDrop {
    fn destroy(&mut self, ctx: &VkCtx);
}

pub struct VkCtx {
    entry: ash::Entry,
    pub instance: ash::Instance,
    pub device: ash::Device,
    pub physical_device: vk::PhysicalDevice,
    pub physical_device_properties: vk::PhysicalDeviceProperties,
    pub surface_loader: ash::khr::surface::Instance,
    pub surface: vk::SurfaceKHR,
    pub queue: vk::Queue,
    pub allocator: vk_mem::Allocator,
    pub command_pool: vk::CommandPool,
    pub command_buffers: Vec<vk::CommandBuffer>,
    pub frame_fences: Vec<vk::Fence>,
    pub image_acquired_semaphores: Vec<vk::Semaphore>,
    setup_cb: vk::CommandBuffer,
}

impl VkCtx {
    pub fn new(window: &winit::window::Window, max_frames_in_flight: u32) -> Self {
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
        if has_device_extension(ash::khr::dynamic_rendering_local_read::NAME) {
            device_extensions.push(ash::khr::dynamic_rendering_local_read::NAME.as_ptr());
        }

        let vk_10_features = vk::PhysicalDeviceFeatures::default().sampler_anisotropy(true);
        let mut vk_11_features =
            vk::PhysicalDeviceVulkan11Features::default().shader_draw_parameters(true);
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
                    &&vk::DeviceCreateInfo::default()
                        .queue_create_infos(&[vk::DeviceQueueCreateInfo {
                            queue_family_index,
                            queue_count: 1,
                            p_queue_priorities: p_queue_priorities.as_ptr(),
                            ..Default::default()
                        }])
                        .enabled_extension_names(&device_extensions)
                        .enabled_features(&vk_10_features)
                        .push_next(&mut vk_13_features)
                        .push_next(&mut vk_12_features)
                        .push_next(&mut vk_11_features),
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
                    command_buffer_count: max_frames_in_flight,
                    ..Default::default()
                })
                .unwrap()
        };

        let mut frame_fences = Vec::new();
        let mut image_acquired_semaphores = Vec::new();

        for _ in 0..max_frames_in_flight {
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

        let setup_cb = unsafe {
            device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(command_pool)
                        .command_buffer_count(1),
                )
                .unwrap()[0]
        };

        Self {
            entry,
            instance,
            device,
            physical_device,
            physical_device_properties,
            surface_loader,
            surface,
            queue,
            allocator,
            command_pool,
            command_buffers,
            frame_fences,
            image_acquired_semaphores,
            setup_cb,
        }
    }

    pub fn destroy<T: VkDrop>(&self, resource: &mut T) {
        resource.destroy(self);
    }

    pub fn with_setup_cb(&self, callback: impl FnOnce(vk::CommandBuffer)) {
        unsafe {
            self.device
                .begin_command_buffer(
                    self.setup_cb,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .unwrap()
        };

        callback(self.setup_cb);

        unsafe {
            self.device.end_command_buffer(self.setup_cb).unwrap();
            self.device
                .queue_submit(
                    self.queue,
                    &[vk::SubmitInfo::default().command_buffers(&[self.setup_cb])],
                    vk::Fence::null(),
                )
                .unwrap();
            self.device.device_wait_idle().unwrap()
        }
    }
}
