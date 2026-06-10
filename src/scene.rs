use ash::vk::{self};
use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Mat4, Quat, Vec3};
use vk_mem::Alloc;

#[derive(Debug)]
pub struct SceneResources {
    pub vertex_buffers: Vec<(vk::Buffer, vk_mem::Allocation)>,
    pub index_buffers: Vec<(vk::Buffer, vk_mem::Allocation)>,
    pub index_counts: Vec<u32>,
    pub images: Vec<(vk::Image, vk_mem::Allocation, vk::ImageView)>,
    pub samplers: Vec<vk::Sampler>,
    pub object_buffer: (vk::Buffer, vk_mem::Allocation, vk::DeviceAddress),
    pub material_buffer: (vk::Buffer, vk_mem::Allocation, vk::DeviceAddress),
    pub primitives: Vec<PrimitiveIndices>,
}

#[derive(Clone, Debug)]
pub struct BoundingBox {
    min: glam::Vec3A,
    max: glam::Vec3A,
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
#[repr(C)]
pub struct PrimitiveIndices {
    pub object_id: u32,
    pub material_id: u32,
}

#[derive(Debug, Clone, Copy)]
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
    pub fn create(
        physical_device_properties: &vk::PhysicalDeviceProperties,
        device: &ash::Device,
        queue: &vk::Queue,
        allocator: &vk_mem::Allocator,
        command_pool: &vk::CommandPool,
    ) -> Self {
        let (document, buffers, textures) = gltf::import("assets/sun_temple.glb").unwrap();
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
        let mut primitives = Vec::new();

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

                primitives.push(PrimitiveIndices {
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
            primitives,
        }
    }
}
