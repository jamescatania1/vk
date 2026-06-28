use ash::vk;

use crate::{
    config::Config,
    utils::{
        context::{VkCtx, VkDrop},
        image::{Image, image},
    },
};

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

pub struct ScreenImages {
    pub depth: Image,
    pub gbuffer: Image,
    pub ao: Image,
    pub ao_history: [Image; 2],
    pub color_output: Image,
}

impl VkDrop for ScreenImages {
    fn destroy(&mut self, ctx: &VkCtx) {
        self.depth.destroy(ctx);
        self.color_output.destroy(ctx);
        self.gbuffer.destroy(ctx);
        self.ao.destroy(ctx);
        self.ao_history[0].destroy(ctx);
        self.ao_history[1].destroy(ctx);
    }
}

impl ScreenImages {
    fn new(ctx: &VkCtx, render_size: vk::Extent2D, render_size_half: vk::Extent2D) -> Self {
        let depth = image()
            .format(DEPTH_FORMAT)
            .extent_2d(render_size.into())
            .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED)
            .create(ctx, vk_mem::AllocationCreateFlags::DEDICATED_MEMORY);

        let gbuffer = image()
            .format(vk::Format::R32G32B32A32_UINT)
            .extent_2d(render_size.into())
            .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED)
            .create(ctx, vk_mem::AllocationCreateFlags::DEDICATED_MEMORY);

        let ao = image()
            .extent_2d(render_size_half)
            .format(vk::Format::R32_SFLOAT)
            .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED)
            .create(ctx, vk_mem::AllocationCreateFlags::DEDICATED_MEMORY);

        let ao_history = [0, 1].map(|_| {
            image()
                .extent_2d(render_size_half)
                .format(vk::Format::R32_SFLOAT)
                .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED)
                .create(ctx, vk_mem::AllocationCreateFlags::DEDICATED_MEMORY)
        });

        let color_output = image()
            .extent_2d(render_size)
            .format(vk::Format::R16G16B16A16_SFLOAT)
            .usage(
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::STORAGE
                    | vk::ImageUsageFlags::SAMPLED,
            )
            .create(ctx, vk_mem::AllocationCreateFlags::DEDICATED_MEMORY);

        Self {
            depth,
            gbuffer,
            ao,
            ao_history,
            color_output,
        }
    }
}

impl ScreenResources {
    pub fn new(ctx: &VkCtx, config: &Config) -> Self {
        let swapchain_loader = ash::khr::swapchain::Device::new(&ctx.instance, &ctx.device);
        let surface_capabilities = unsafe {
            ctx.surface_loader
                .get_physical_device_surface_capabilities(ctx.physical_device, ctx.surface)
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
                        .surface(ctx.surface)
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
                    ctx.device
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
            render_complete_semaphores.push(unsafe {
                ctx.device
                    .create_semaphore(&Default::default(), None)
                    .unwrap()
            });
        }

        let images = ScreenImages::new(ctx, render_size, render_size_half);

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

    pub fn update_descriptors(&mut self, ctx: &VkCtx, descriptor_set: vk::DescriptorSet) {
        let ao_history_infos = self
            .images
            .ao_history
            .iter_mut()
            .map(|img| img.info_default(ctx, vk::ImageLayout::GENERAL))
            .collect::<Vec<_>>();

        unsafe {
            ctx.device.update_descriptor_sets(
                &[
                    vk::WriteDescriptorSet::default()
                        .dst_set(descriptor_set)
                        .dst_binding(5)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                        .image_info(&[self
                            .images
                            .depth
                            .info_default(ctx, vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL)]),
                    vk::WriteDescriptorSet::default()
                        .dst_set(descriptor_set)
                        .dst_binding(6)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                        .image_info(&[self
                            .images
                            .gbuffer
                            .info_default(ctx, vk::ImageLayout::READ_ONLY_OPTIMAL)]),
                    vk::WriteDescriptorSet::default()
                        .dst_set(descriptor_set)
                        .dst_binding(7)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .image_info(&[self
                            .images
                            .color_output
                            .info_default(ctx, vk::ImageLayout::GENERAL)]),
                    vk::WriteDescriptorSet::default()
                        .dst_set(descriptor_set)
                        .dst_binding(8)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                        .image_info(&[self
                            .images
                            .color_output
                            .info_default(ctx, vk::ImageLayout::READ_ONLY_OPTIMAL)]),
                    vk::WriteDescriptorSet::default()
                        .dst_set(descriptor_set)
                        .dst_binding(9)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .image_info(&[self.images.ao.info_default(ctx, vk::ImageLayout::GENERAL)]),
                    vk::WriteDescriptorSet::default()
                        .dst_set(descriptor_set)
                        .dst_binding(10)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                        .image_info(&[self
                            .images
                            .ao
                            .info_default(ctx, vk::ImageLayout::READ_ONLY_OPTIMAL)]),
                    vk::WriteDescriptorSet::default()
                        .dst_set(descriptor_set)
                        .dst_binding(11)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .image_info(&ao_history_infos),
                    vk::WriteDescriptorSet::default()
                        .dst_set(descriptor_set)
                        .dst_binding(12)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                        .image_info(&ao_history_infos),
                ],
                &[],
            )
        };
    }

    pub fn recreate(&mut self, ctx: &VkCtx, cfg: &Config, descriptor_set: vk::DescriptorSet) {
        self.recreate = false;
        unsafe {
            ctx.device.device_wait_idle().unwrap();
        };
        let old_swapchain = self.swapchain;

        for sem in self.render_complete_semaphores.drain(..) {
            unsafe { ctx.device.destroy_semaphore(sem, None) };
        }
        for view in self.swapchain_views.drain(..) {
            unsafe { ctx.device.destroy_image_view(view, None) };
        }

        self.images.destroy(&ctx);
        // self.images.destroy(&ctx.device, &ctx.allocator);

        let surface_capabilities = unsafe {
            ctx.surface_loader
                .get_physical_device_surface_capabilities(ctx.physical_device, ctx.surface)
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
                        .surface(ctx.surface)
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
                    ctx.device
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

        self.images = ScreenImages::new(ctx, self.render_size, self.render_size_half);

        self.render_complete_semaphores = Vec::new();
        for _ in 0..self.swapchain_images.len() {
            self.render_complete_semaphores.push(unsafe {
                ctx.device
                    .create_semaphore(&Default::default(), None)
                    .unwrap()
            });
        }

        self.update_descriptors(ctx, descriptor_set);
    }
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
