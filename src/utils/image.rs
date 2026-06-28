use std::collections::HashMap;

use ash::vk;
use vk_mem::Alloc;

use crate::utils::{
    buffer::Buffer,
    context::{VkCtx, VkDrop},
};

pub fn image() -> ImageDesc {
    ImageDesc::default()
}

pub struct ImageDesc {
    flags: vk::ImageCreateFlags,
    image_type: vk::ImageType,
    format: vk::Format,
    extent: vk::Extent3D,
    samples: vk::SampleCountFlags,
    mip_levels: u32,
    array_layers: u32,
    usage: vk::ImageUsageFlags,
}

impl Default for ImageDesc {
    fn default() -> Self {
        Self {
            flags: Default::default(),
            image_type: Default::default(),
            format: Default::default(),
            extent: Default::default(),
            samples: vk::SampleCountFlags::TYPE_1,
            mip_levels: 1,
            array_layers: 1,
            usage: Default::default(),
        }
    }
}

impl ImageDesc {
    pub fn flags(mut self, flags: vk::ImageCreateFlags) -> Self {
        self.flags = flags;
        self
    }

    pub fn format(mut self, format: vk::Format) -> Self {
        self.format = format;
        self
    }

    pub fn extent_1d(mut self, extent: u32) -> Self {
        self.image_type = vk::ImageType::TYPE_1D;
        self.extent = vk::Extent3D {
            width: extent,
            height: 1,
            depth: 1,
        };
        self.array_layers = 1;
        self
    }
    pub fn extent_2d(mut self, extent: vk::Extent2D) -> Self {
        self.image_type = vk::ImageType::TYPE_2D;
        self.extent = extent.into();
        self.array_layers = 1;
        self
    }
    pub fn extent_2d_array(mut self, extent: vk::Extent2D, array_layers: u32) -> Self {
        self.image_type = vk::ImageType::TYPE_2D;
        self.extent = extent.into();
        self.array_layers = array_layers;
        self
    }
    pub fn extent_3d(mut self, extent: vk::Extent3D) -> Self {
        self.image_type = vk::ImageType::TYPE_3D;
        self.extent = extent;
        self.array_layers = 1;
        self
    }

    pub fn samples(mut self, samples: vk::SampleCountFlags) -> Self {
        self.samples = samples;
        self
    }

    pub fn mip_levels(mut self, mip_levels: u32) -> Self {
        self.mip_levels = mip_levels;
        self
    }

    pub fn usage(mut self, usage: vk::ImageUsageFlags) -> Self {
        self.usage = usage;
        self
    }

    pub fn create(self, ctx: &VkCtx, flags: vk_mem::AllocationCreateFlags) -> Image {
        let (image, allocation) = unsafe {
            ctx.allocator
                .create_image(
                    &vk::ImageCreateInfo {
                        image_type: self.image_type,
                        format: self.format,
                        extent: self.extent,
                        mip_levels: self.mip_levels,
                        array_layers: self.array_layers,
                        samples: self.samples,
                        tiling: vk::ImageTiling::OPTIMAL,
                        usage: self.usage,
                        initial_layout: vk::ImageLayout::UNDEFINED,
                        ..Default::default()
                    },
                    &vk_mem::AllocationCreateInfo {
                        flags,
                        usage: vk_mem::MemoryUsage::Auto,
                        ..Default::default()
                    },
                )
                .unwrap()
        };
        let default_view_desc = ViewDesc::default_from_image(&self);
        Image {
            image,
            allocation,
            desc: self,
            default_view_desc,
            views: Default::default(),
        }
    }

    pub fn create_with_data(
        self,
        ctx: &VkCtx,
        flags: vk_mem::AllocationCreateFlags,
        bytes: &[u8],
    ) -> Image {
        let mut staging = Buffer::new(
            ctx,
            bytes.len() as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk_mem::MemoryUsage::Auto,
            vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
                | vk_mem::AllocationCreateFlags::MAPPED,
        );
        staging.write(ctx, bytes);

        let (image, allocation) = unsafe {
            ctx.allocator
                .create_image(
                    &vk::ImageCreateInfo {
                        image_type: self.image_type,
                        format: self.format,
                        extent: self.extent,
                        mip_levels: self.mip_levels,
                        array_layers: self.array_layers,
                        samples: self.samples,
                        tiling: vk::ImageTiling::OPTIMAL,
                        usage: self.usage | vk::ImageUsageFlags::TRANSFER_DST,
                        initial_layout: vk::ImageLayout::UNDEFINED,
                        ..Default::default()
                    },
                    &vk_mem::AllocationCreateInfo {
                        flags,
                        usage: vk_mem::MemoryUsage::Auto,
                        ..Default::default()
                    },
                )
                .unwrap()
        };

        let default_view_desc = ViewDesc::default_from_image(&self);

        ctx.with_setup_cb(|cb| unsafe {
            ctx.device.cmd_pipeline_barrier2(
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
                                .aspect_mask(default_view_desc.aspect_mask)
                                .base_mip_level(0)
                                .level_count(self.mip_levels)
                                .layer_count(1),
                        ),
                ]),
            );
            ctx.device.cmd_copy_buffer_to_image(
                cb,
                staging.buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::BufferImageCopy::default()
                    .buffer_offset(0)
                    .image_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(default_view_desc.aspect_mask)
                            .mip_level(0)
                            .layer_count(1),
                    )
                    .image_extent(self.extent)],
            );
        });

        staging.destroy(ctx);

        Image {
            image,
            allocation,
            desc: self,
            default_view_desc,
            views: Default::default(),
        }
    }
}

#[derive(Clone, Copy, Eq, PartialEq, Hash)]
pub struct ViewDesc {
    pub view_type: vk::ImageViewType,
    pub aspect_mask: vk::ImageAspectFlags,
    pub base_mip_level: u32,
    pub level_count: u32,
    pub base_array_layer: u32,
    pub layer_count: u32,
}

impl ViewDesc {
    fn default_from_image(desc: &ImageDesc) -> Self {
        Self {
            view_type: match (desc.image_type, desc.array_layers) {
                (vk::ImageType::TYPE_1D, _) => vk::ImageViewType::TYPE_1D,
                (vk::ImageType::TYPE_2D, 1) => vk::ImageViewType::TYPE_2D,
                (vk::ImageType::TYPE_2D, _) => vk::ImageViewType::TYPE_2D_ARRAY,
                (vk::ImageType::TYPE_3D, _) => vk::ImageViewType::TYPE_3D,
                _ => vk::ImageViewType::TYPE_2D,
            },
            aspect_mask: match desc.format {
                vk::Format::D16_UNORM | vk::Format::D32_SFLOAT => vk::ImageAspectFlags::DEPTH,
                vk::Format::D16_UNORM_S8_UINT
                | vk::Format::D24_UNORM_S8_UINT
                | vk::Format::D32_SFLOAT_S8_UINT => {
                    vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
                }
                _ => vk::ImageAspectFlags::COLOR,
            },
            base_mip_level: 0,
            level_count: desc.mip_levels,
            base_array_layer: 0,
            layer_count: desc.array_layers,
        }
    }
}

pub struct Image {
    pub image: vk::Image,
    pub allocation: vk_mem::Allocation,
    desc: ImageDesc,
    default_view_desc: ViewDesc,
    views: HashMap<ViewDesc, vk::ImageView>,
}

impl VkDrop for Image {
    fn destroy(&mut self, ctx: &VkCtx) {
        unsafe {
            for view in self.views.values() {
                ctx.device.destroy_image_view(*view, None);
            }
            ctx.allocator
                .destroy_image(self.image, &mut self.allocation);
        };
    }
}

impl Image {
    fn view(&mut self, ctx: &VkCtx, desc: ViewDesc) -> vk::ImageView {
        if let Some(view) = self.views.get(&desc) {
            *view
        } else {
            let view = unsafe {
                ctx.device
                    .create_image_view(
                        &vk::ImageViewCreateInfo {
                            image: self.image,
                            view_type: desc.view_type,
                            format: self.desc.format,
                            subresource_range: vk::ImageSubresourceRange {
                                aspect_mask: desc.aspect_mask,
                                base_mip_level: desc.base_mip_level,
                                level_count: desc.level_count,
                                base_array_layer: desc.base_array_layer,
                                layer_count: desc.layer_count,
                            },
                            ..Default::default()
                        },
                        None,
                    )
                    .unwrap()
            };
            self.views.insert(desc, view);
            view
        }
    }

    pub fn view_default(&mut self, ctx: &VkCtx) -> vk::ImageView {
        self.view(ctx, self.default_view_desc)
    }
    pub fn view_array_layer(&mut self, ctx: &VkCtx, layer: u32) -> vk::ImageView {
        self.view(
            ctx,
            ViewDesc {
                view_type: vk::ImageViewType::TYPE_2D,
                base_array_layer: layer,
                layer_count: 1,
                ..self.default_view_desc
            },
        )
    }
    pub fn view_mip_level(&mut self, ctx: &VkCtx, mip_level: u32) -> vk::ImageView {
        self.view(
            ctx,
            ViewDesc {
                base_mip_level: mip_level,
                level_count: 1,
                ..self.default_view_desc
            },
        )
    }

    pub fn info_default(
        &mut self,
        ctx: &VkCtx,
        layout: vk::ImageLayout,
    ) -> vk::DescriptorImageInfo {
        vk::DescriptorImageInfo::default()
            .image_view(self.view_default(ctx))
            .image_layout(layout)
    }
    pub fn info_array_layer(
        &mut self,
        ctx: &VkCtx,
        layout: vk::ImageLayout,
        array_layer: u32,
    ) -> vk::DescriptorImageInfo {
        vk::DescriptorImageInfo::default()
            .image_view(self.view_array_layer(ctx, array_layer))
            .image_layout(layout)
    }
    pub fn info_mip_level(
        &mut self,
        ctx: &VkCtx,
        layout: vk::ImageLayout,
        mip_level: u32,
    ) -> vk::DescriptorImageInfo {
        vk::DescriptorImageInfo::default()
            .image_view(self.view_mip_level(ctx, mip_level))
            .image_layout(layout)
    }

    // delete later, simple blit isn't really good for scene textures
    pub fn generate_mipmaps(&mut self, ctx: &VkCtx, target_layout: vk::ImageLayout) {
        let mut w = self.desc.extent.width;
        let mut h = self.desc.extent.height;

        ctx.with_setup_cb(|cb| unsafe {
            for i in 1..self.desc.mip_levels {
                ctx.device.cmd_pipeline_barrier2(
                    cb,
                    &vk::DependencyInfo::default().image_memory_barriers(&[
                        vk::ImageMemoryBarrier2::default()
                            .image(self.image)
                            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                            .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                            .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                            .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                            .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                            .subresource_range(
                                vk::ImageSubresourceRange::default()
                                    .aspect_mask(self.default_view_desc.aspect_mask)
                                    .base_mip_level(i - 1)
                                    .level_count(1)
                                    .base_array_layer(0)
                                    .layer_count(1),
                            ),
                    ]),
                );

                let dst_w = (w / 2).max(1);
                let dst_h = (h / 2).max(1);

                ctx.device.cmd_blit_image(
                    cb,
                    self.image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    self.image,
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
                                .aspect_mask(self.default_view_desc.aspect_mask)
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
                                .aspect_mask(self.default_view_desc.aspect_mask)
                                .mip_level(i)
                                .base_array_layer(0)
                                .layer_count(1),
                        )],
                    vk::Filter::LINEAR,
                );

                ctx.device.cmd_pipeline_barrier2(
                    cb,
                    &vk::DependencyInfo::default().image_memory_barriers(&[
                        vk::ImageMemoryBarrier2::default()
                            .image(self.image)
                            .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                            .src_access_mask(vk::AccessFlags2::TRANSFER_READ)
                            .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                            .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                            .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                            .new_layout(target_layout)
                            .subresource_range(
                                vk::ImageSubresourceRange::default()
                                    .aspect_mask(self.default_view_desc.aspect_mask)
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

            ctx.device.cmd_pipeline_barrier2(
                cb,
                &vk::DependencyInfo::default().image_memory_barriers(&[
                    vk::ImageMemoryBarrier2::default()
                        .image(self.image)
                        .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                        .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                        .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .new_layout(target_layout)
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(self.default_view_desc.aspect_mask)
                                .base_mip_level(self.desc.mip_levels - 1)
                                .level_count(1)
                                .base_array_layer(0)
                                .layer_count(1),
                        ),
                ]),
            );
        });
    }
}
