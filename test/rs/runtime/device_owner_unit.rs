/// A resource must retain the complete device domain after its Engine is gone,
/// and the final resource release must not leave an ownership cycle.
#[test]
fn buffer_retains_device_after_engine_drop() -> crate::Result<()> {
	let engine = match crate::Engine::new() {
		Ok(engine) => engine,
		Err(error)
			if matches!(
				error.kind(),
				crate::ErrorKind::NoSuitableDevice | crate::ErrorKind::BackendUnavailable
			) =>
		{
			return Ok(());
		}
		Err(error) => return Err(error),
	};
	let device = engine.handle().state.borrow().device.clone();
	let weak = std::sync::Arc::downgrade(&device);
	let buffer = super::Buffer::host_visible_storage(&device, 4)?;
	buffer.write(0, &[3, 1, 4, 1])?;
	drop(device);
	drop(engine);
	assert!(weak.upgrade().is_some(), "buffer must retain its Device");
	let mut observed = [0; 4];
	buffer.read(0, &mut observed)?;
	assert_eq!(observed, [3, 1, 4, 1]);
	drop(buffer);
	assert!(
		weak.upgrade().is_none(),
		"Device must have no ownership cycle"
	);
	Ok(())
}
