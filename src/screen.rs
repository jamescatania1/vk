use ash::vk;
use vk_mem::Alloc;

use crate::config::Config;

pub struct ScreenResources {
    pub recreate: bool,
    pub swapchain_loader: ash::khr::swapchain::Device,
    pub swapchain: vk::SwapchainKHR,
    pub swapchain_images: Vec<vk::Image>,
    pub swapchain_views: Vec<vk::ImageView>,
    pub render_complete_semaphores: Vec<vk::Semaphore>,
    pub images: ScreenImages,
    pub viewport_size: vk::Extent2D,
    pub render_size: vk::Extent2D,
    pub texel_size: glam::Vec2,
    pub render_size_half: vk::Extent2D,
    pub texel_size_half: glam::Vec2,
}

const SWAPCHAIN_FORMAT: vk::Format = vk::Format::B8G8R8A8_SRGB;
const DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;

pub struct Image {
    pub image: vk::Image,
    pub allocation: vk_mem::Allocation,
    pub view: vk::ImageView,
}

pub struct ScreenImages {
    pub depth: Image,
    pub depth_half: Image,
    pub gbuffer: Image,
    pub ao: Image,
    pub color_output: Image,
}

pub trait ExtentExt {
    fn mul_ceil(self, x: f32) -> Self;
    fn as_uvec2(self) -> glam::UVec2;
}
impl ExtentExt for vk::Extent2D {
    fn mul_ceil(self, x: f32) -> Self {
        vk::Extent2D {
            width: (self.width as f32 * x).ceil() as u32,
            height: (self.height as f32 * x).ceil() as u32,
        }
    }
    fn as_uvec2(self) -> glam::UVec2 {
        glam::uvec2(self.width, self.height)
    }
}

impl ScreenImages {
    fn new(
        device: &ash::Device,
        allocator: &vk_mem::Allocator,
        render_size: vk::Extent2D,
        render_size_half: vk::Extent2D,
    ) -> Self {
        let depth = unsafe {
            let (image, allocation) = allocator
                .create_image(
                    &vk::ImageCreateInfo::default()
                        .image_type(vk::ImageType::TYPE_2D)
                        .format(DEPTH_FORMAT)
                        .extent(render_size.into())
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
            Image {
                image,
                allocation,
                view,
            }
        };

        let depth_half = unsafe {
            let (image, allocation) = allocator
                .create_image(
                    &vk::ImageCreateInfo::default()
                        .image_type(vk::ImageType::TYPE_2D)
                        .format(vk::Format::R32_SFLOAT)
                        .extent(render_size_half.into())
                        .mip_levels(1)
                        .array_layers(1)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .tiling(vk::ImageTiling::OPTIMAL)
                        .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED)
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
                        format: vk::Format::R32_SFLOAT,
                        subresource_range: vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .level_count(1)
                            .layer_count(1),
                        ..Default::default()
                    },
                    None,
                )
                .unwrap();
            Image {
                image,
                allocation,
                view,
            }
        };

        let gbuffer = unsafe {
            let (image, allocation) = allocator
                .create_image(
                    &vk::ImageCreateInfo::default()
                        .image_type(vk::ImageType::TYPE_2D)
                        .format(vk::Format::R32G32B32A32_UINT)
                        .extent(render_size.into())
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
                        format: vk::Format::R32G32B32A32_UINT,
                        subresource_range: vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .level_count(1)
                            .layer_count(1),
                        ..Default::default()
                    },
                    None,
                )
                .unwrap();
            Image {
                image,
                allocation,
                view,
            }
        };

        let ao = unsafe {
            let (image, allocation) = allocator
                .create_image(
                    &vk::ImageCreateInfo::default()
                        .image_type(vk::ImageType::TYPE_2D)
                        .format(vk::Format::R32_SFLOAT)
                        .extent(render_size_half.into())
                        .mip_levels(1)
                        .array_layers(1)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .tiling(vk::ImageTiling::OPTIMAL)
                        .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED)
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
                        format: vk::Format::R32_SFLOAT,
                        subresource_range: vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .level_count(1)
                            .layer_count(1),
                        ..Default::default()
                    },
                    None,
                )
                .unwrap();
            Image {
                image,
                allocation,
                view,
            }
        };

        let color_output = unsafe {
            let (image, allocation) = allocator
                .create_image(
                    &vk::ImageCreateInfo::default()
                        .image_type(vk::ImageType::TYPE_2D)
                        .format(vk::Format::R16G16B16A16_SFLOAT)
                        .extent(render_size.into())
                        .mip_levels(1)
                        .array_layers(1)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .tiling(vk::ImageTiling::OPTIMAL)
                        .usage(
                            vk::ImageUsageFlags::COLOR_ATTACHMENT
                                | vk::ImageUsageFlags::STORAGE
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
            Image {
                image,
                allocation,
                view,
            }
        };

        Self {
            depth,
            depth_half,
            gbuffer,
            ao,
            color_output,
        }
    }

    fn destroy(&mut self, device: &ash::Device, allocator: &vk_mem::Allocator) {
        unsafe {
            device.destroy_image_view(self.depth.view, None);
            allocator.destroy_image(self.depth.image, &mut self.depth.allocation);

            device.destroy_image_view(self.depth_half.view, None);
            allocator.destroy_image(self.depth_half.image, &mut self.depth_half.allocation);

            device.destroy_image_view(self.color_output.view, None);
            allocator.destroy_image(self.color_output.image, &mut self.color_output.allocation);

            device.destroy_image_view(self.gbuffer.view, None);
            allocator.destroy_image(self.gbuffer.image, &mut self.gbuffer.allocation);

            device.destroy_image_view(self.ao.view, None);
            allocator.destroy_image(self.ao.image, &mut self.ao.allocation);
        }
    }
}

impl ScreenResources {
    pub fn new(
        instance: &ash::Instance,
        device: &ash::Device,
        allocator: &vk_mem::Allocator,
        physical_device: vk::PhysicalDevice,
        surface_loader: &ash::khr::surface::Instance,
        surface: vk::SurfaceKHR,
        config: &Config,
    ) -> Self {
        let swapchain_loader = ash::khr::swapchain::Device::new(&instance, &device);
        let surface_capabilities = unsafe {
            surface_loader
                .get_physical_device_surface_capabilities(physical_device, surface)
                .unwrap()
        };

        let viewport_size = surface_capabilities.current_extent;

        let render_size = viewport_size.mul_ceil(config.render_scale);
        let texel_size = 1.0 / render_size.as_uvec2().as_vec2();

        let render_size_half = viewport_size.mul_ceil(0.5);
        let texel_size_half = 1.0 / render_size_half.as_uvec2().as_vec2();

        let (swapchain, swapchain_images, swapchain_views) = unsafe {
            let swapchain = swapchain_loader
                .create_swapchain(
                    &vk::SwapchainCreateInfoKHR::default()
                        .surface(surface)
                        .min_image_count(surface_capabilities.min_image_count)
                        .image_format(SWAPCHAIN_FORMAT)
                        .image_color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR)
                        .image_extent(viewport_size)
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

        let mut render_complete_semaphores = Vec::new();
        for _ in 0..swapchain_images.len() {
            render_complete_semaphores
                .push(unsafe { device.create_semaphore(&Default::default(), None).unwrap() });
        }

        let images = ScreenImages::new(device, allocator, render_size, render_size_half);

        Self {
            recreate: false,
            swapchain_loader,
            swapchain,
            swapchain_images,
            swapchain_views,
            render_complete_semaphores,
            images,
            viewport_size,
            render_size,
            texel_size,
            render_size_half,
            texel_size_half,
        }
    }

    pub fn update_descriptors(&self, device: &ash::Device, descriptor_set: vk::DescriptorSet) {
        let depth_info = vk::DescriptorImageInfo::default()
            .image_view(self.images.depth.view)
            .image_layout(vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL);
        let gbuffer_info = vk::DescriptorImageInfo::default()
            .image_view(self.images.gbuffer.view)
            .image_layout(vk::ImageLayout::READ_ONLY_OPTIMAL);
        let color_output_info_storage = vk::DescriptorImageInfo::default()
            .image_view(self.images.color_output.view)
            .image_layout(vk::ImageLayout::GENERAL);
        let color_output_info = vk::DescriptorImageInfo::default()
            .image_view(self.images.color_output.view)
            .image_layout(vk::ImageLayout::READ_ONLY_OPTIMAL);
        let ao_info_storage = vk::DescriptorImageInfo::default()
            .image_view(self.images.ao.view)
            .image_layout(vk::ImageLayout::GENERAL);
        let ao_info = vk::DescriptorImageInfo::default()
            .image_view(self.images.ao.view)
            .image_layout(vk::ImageLayout::READ_ONLY_OPTIMAL);
        let depth_half_info_storage = vk::DescriptorImageInfo::default()
            .image_view(self.images.depth_half.view)
            .image_layout(vk::ImageLayout::GENERAL);
        let depth_half_info = vk::DescriptorImageInfo::default()
            .image_view(self.images.depth_half.view)
            .image_layout(vk::ImageLayout::READ_ONLY_OPTIMAL);
        unsafe {
            device.update_descriptor_sets(
                &[
                    vk::WriteDescriptorSet::default()
                        .dst_set(descriptor_set)
                        .dst_binding(3)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                        .image_info(&[depth_info]),
                    vk::WriteDescriptorSet::default()
                        .dst_set(descriptor_set)
                        .dst_binding(4)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                        .image_info(&[gbuffer_info]),
                    vk::WriteDescriptorSet::default()
                        .dst_set(descriptor_set)
                        .dst_binding(5)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .image_info(&[color_output_info_storage]),
                    vk::WriteDescriptorSet::default()
                        .dst_set(descriptor_set)
                        .dst_binding(6)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                        .image_info(&[color_output_info]),
                    vk::WriteDescriptorSet::default()
                        .dst_set(descriptor_set)
                        .dst_binding(8)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .image_info(&[ao_info_storage]),
                    vk::WriteDescriptorSet::default()
                        .dst_set(descriptor_set)
                        .dst_binding(9)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                        .image_info(&[ao_info]),
                    vk::WriteDescriptorSet::default()
                        .dst_set(descriptor_set)
                        .dst_binding(10)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .image_info(&[depth_half_info_storage]),
                    vk::WriteDescriptorSet::default()
                        .dst_set(descriptor_set)
                        .dst_binding(11)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                        .image_info(&[depth_half_info]),
                ],
                &[],
            )
        };
    }

    pub fn recreate(
        &mut self,
        cfg: &Config,
        device: &ash::Device,
        allocator: &vk_mem::Allocator,
        physical_device: vk::PhysicalDevice,
        surface_loader: &ash::khr::surface::Instance,
        surface: vk::SurfaceKHR,
        descriptor_set: vk::DescriptorSet,
    ) {
        self.recreate = false;
        unsafe {
            device.device_wait_idle().unwrap();
        };
        let old_swapchain = self.swapchain;

        for sem in self.render_complete_semaphores.drain(..) {
            unsafe { device.destroy_semaphore(sem, None) };
        }
        for view in self.swapchain_views.drain(..) {
            unsafe { device.destroy_image_view(view, None) };
        }

        self.images.destroy(&device, &allocator);

        let surface_capabilities = unsafe {
            surface_loader
                .get_physical_device_surface_capabilities(physical_device, surface)
                .unwrap()
        };

        self.viewport_size = surface_capabilities.current_extent;

        self.render_size = self.viewport_size.mul_ceil(cfg.render_scale);
        self.texel_size = 1.0 / self.render_size.as_uvec2().as_vec2();

        self.render_size_half = self.render_size.mul_ceil(0.5);
        self.texel_size_half = 1.0 / self.render_size_half.as_uvec2().as_vec2();

        self.swapchain = unsafe {
            self.swapchain_loader
                .create_swapchain(
                    &vk::SwapchainCreateInfoKHR::default()
                        .surface(surface)
                        .min_image_count(surface_capabilities.min_image_count)
                        .image_format(SWAPCHAIN_FORMAT)
                        .image_color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR)
                        .image_extent(self.viewport_size)
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

        (self.swapchain_images, self.swapchain_views) = unsafe {
            let images = self
                .swapchain_loader
                .get_swapchain_images(self.swapchain)
                .unwrap();
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
            (images, views)
        };

        self.images =
            ScreenImages::new(&device, &allocator, self.render_size, self.render_size_half);

        self.render_complete_semaphores = Vec::new();
        for _ in 0..self.swapchain_images.len() {
            self.render_complete_semaphores
                .push(unsafe { device.create_semaphore(&Default::default(), None).unwrap() });
        }

        self.update_descriptors(device, descriptor_set);
    }
}
