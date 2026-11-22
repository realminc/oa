use super::*;

#[test]
#[ignore = "requires an admitted Vulkan device"]
fn partial_batch_submission_retires_accepted_work_and_releases_the_tail() -> Result<()> {
	let engine = Engine::new()?;
	let owner = Arc::new(());
	let accepted;
	{
		let mut state = engine.handle.state.borrow_mut();
		let start = state.next_epoch;
		let mut batch = RecordedBatch::new(&state.device);
		for _ in 0..3 {
			batch.push(state.device.record_empty()?);
		}
		batch.retain(owner.clone());
		assert_eq!(Arc::strong_count(&owner), 4);

		let mut first = None;
		let result = submit_recorded_batch_with(&mut state, batch, false, |state, command| {
			if first.is_some() {
				state.device.free(command);
				return Err(Error::internal("test rejected the second queue submission"));
			}
			let event = submit_recorded(state, command)?;
			first = Some(event.clone());
			Ok(event)
		});
		assert!(result.unwrap_err().to_string().contains("test rejected"));
		accepted = first.expect("first command was submitted");
		assert_eq!(accepted.epoch(), start + 1);
		assert_eq!(state.next_epoch, accepted.epoch());
	}
	accepted.wait()?;
	assert_eq!(Arc::strong_count(&owner), 1);
	Ok(())
}
