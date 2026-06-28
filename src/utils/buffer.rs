use ash::vk;
use vk_mem::Alloc;

use crate::utils::context::{VkCtx, VkDrop};

#[derive(Debug)]
pub struct Buffer {
    pub buffer: vk::Buffer,
    pub allocation: vk_mem::Allocation,
    size: u64,
    address: Option<vk::DeviceAddress>,
}

impl VkDrop for Buffer {
    fn destroy(&mut self, ctx: &VkCtx) {
        unsafe {
            ctx.allocator
                .destroy_buffer(self.buffer, &mut self.allocation)
        };
    }
}

impl Buffer {
    pub fn new(
        ctx: &VkCtx,
        size: u64,
        usage: vk::BufferUsageFlags,
        mem_usage: vk_mem::MemoryUsage,
        mem_flags: vk_mem::AllocationCreateFlags,
    ) -> Self {
        let (buffer, allocation) = unsafe {
            ctx.allocator
                .create_buffer(
                    &vk::BufferCreateInfo {
                        size,
                        usage,
                        ..Default::default()
                    },
                    &vk_mem::AllocationCreateInfo {
                        flags: mem_flags,
                        usage: mem_usage,
                        ..Default::default()
                    },
                )
                .unwrap()
        };
        Self {
            buffer,
            allocation,
            size,
            address: None,
        }
    }

    pub fn write(&self, ctx: &VkCtx, bytes: &[u8]) {
        let alloc_info = ctx.allocator.get_allocation_info(&self.allocation);

        let ptr = alloc_info.mapped_data;
        if ptr.is_null() {
            panic!("buffer is unmapped: {:#?}", self);
        }

        let dst = unsafe { std::slice::from_raw_parts_mut(ptr as *mut u8, bytes.len()) };
        dst.copy_from_slice(bytes);
    }

    pub fn flush(&self, ctx: &VkCtx) {
        ctx.allocator
            .flush_allocation(&self.allocation, 0, self.size)
            .unwrap();
    }

    pub fn address(&mut self, ctx: &VkCtx) -> vk::DeviceAddress {
        if self.address.is_none() {
            self.address = unsafe {
                Some(ctx.device.get_buffer_device_address(
                    &vk::BufferDeviceAddressInfo::default().buffer(self.buffer),
                ))
            };
        }
        self.address.unwrap()
    }

    pub fn from_data(ctx: &VkCtx, usage: vk::BufferUsageFlags, bytes: &[u8]) -> Self {
        let mut staging = Self::new(
            ctx,
            bytes.len() as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk_mem::MemoryUsage::Auto,
            vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
                | vk_mem::AllocationCreateFlags::MAPPED,
        );
        staging.write(ctx, bytes);

        let res = Self::new(
            ctx,
            bytes.len() as u64,
            usage | vk::BufferUsageFlags::TRANSFER_DST,
            vk_mem::MemoryUsage::AutoPreferDevice,
            vk_mem::AllocationCreateFlags::empty(),
        );

        ctx.with_setup_cb(|cb| unsafe {
            ctx.device.cmd_copy_buffer(
                cb,
                staging.buffer,
                res.buffer,
                &[vk::BufferCopy::default()
                    .src_offset(0)
                    .dst_offset(0)
                    .size(bytes.len() as u64)],
            );
        });

        ctx.destroy(&mut staging);

        res
    }
}
