//! vkrust consumer proof for the binding migration.
//!
//! Exercises the Stage-1 owned device/allocator lane through the external
//! `vk` crate on real hardware: entry, instance, physical-device selection,
//! logical device, retained synchronized allocator, allocation and ordered
//! teardown. The same native bindings and allocator serve the production backend.

fn describe(error: impl std::fmt::Debug) -> String {
	format!("{error:?}")
}

/// Build the compute lane through vkrust alone.
///
/// # Safety
/// All create-info records, queries and destruction order here satisfy Vulkan
/// valid usage; children are destroyed before the device and instance.
unsafe fn retained_lane() -> Result<(), String> {
	let entry = unsafe { vk::Entry::new() }.map_err(describe)?;
	let application = vk::ApplicationInfo {
		api_version: vk::API_VERSION_1_2,
		..Default::default()
	};
	let instance = unsafe {
		vk::Instance::new(
			&entry,
			&vk::InstanceCreateInfo {
				p_application_info: &application,
				..Default::default()
			},
			None,
		)
	}
	.map_err(describe)?;
	let (physical, family) = unsafe { instance.enumerate_physical_devices() }
		.map_err(describe)?
		.into_iter()
		.find_map(|device| {
			unsafe { instance.get_physical_device_queue_family_properties(device) }
				.ok()?
				.iter()
				.position(|queue| {
					queue.queue_count > 0
						&& queue.queue_flags.as_raw() & vk::QueueFlagBits::COMPUTE_BIT.as_raw() != 0
				})
				.map(|family| (device, family as u32))
		})
		.ok_or("no compute-capable physical device")?;
	let priority = 1.0;
	let queues = vk::DeviceQueueCreateInfo {
		queue_family_index: family,
		queue_count: 1,
		p_queue_priorities: &priority,
		..Default::default()
	};
	let device = std::sync::Arc::new(
		unsafe {
			vk::Device::new(
				&instance,
				physical,
				&vk::DeviceCreateInfo {
					queue_create_info_count: 1,
					p_queue_create_infos: &queues,
					..Default::default()
				},
				None,
			)
		}
		.map_err(describe)?,
	);
	// SAFETY: physical was enumerated from this instance and created this device.
	let allocator = unsafe {
		vk::mem::Allocator::new_retained_from_physical_device(&instance, physical, device.clone())
	}
	.map_err(describe)?;
	let shared = vk::mem::SynchronizedAllocator::new_owned(allocator);
	// SAFETY: synchronized exclusive access; no GPU work is submitted.
	let mut allocation = unsafe {
		shared.with_allocator(|allocator| {
			allocator.allocate(&vk::mem::AllocRequest::new(4096, 1, u32::MAX))
		})
	}
	.map_err(describe)?
	.map_err(describe)?;
	assert!(!allocation.handle().is_null());
	// SAFETY: exact ordered teardown; allocator release precedes device/instance.
	unsafe { shared.with_allocator(|allocator| allocator.free(&mut allocation)) }
		.map_err(describe)?
		.map_err(describe)?;
	drop(shared);
	let device =
		std::sync::Arc::into_inner(device).ok_or("device still shared after allocator teardown")?;
	// SAFETY: no children remain; callbacks match creation (None).
	unsafe { device.destroy(None) };
	unsafe { instance.destroy(None) };
	Ok(())
}

#[test]
#[ignore = "requires a hardware Vulkan 1.2 compute device"]
fn vkrust_owned_allocator_lane_on_hardware() {
	// SAFETY: the lane establishes all native validity itself.
	unsafe { retained_lane() }.expect("vkrust retained allocator lane");
}
