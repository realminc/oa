use super::*;
use crate::{
	DType, Engine, ErrorKind,
	runtime::{BufferBinding, KernelId},
};

#[test]
#[ignore = "requires a hardware Vulkan compute device"]
fn split_recording_requires_an_earlier_exhaustive_producer() -> Result<()> {
	let engine = Engine::new()?;
	let source = Matrix::from_slice(&engine, [1, 32], &[0_u8; 32])?;
	let intermediate = Matrix::allocate_output(source.engine_handle(), vec![1, 32], 32, DType::U8)?;
	let writes = [BufferBinding::write(intermediate.storage())];
	let reads = [BufferBinding::read(intermediate.storage())];
	let updates = [BufferBinding::read_write(intermediate.storage())];
	let aliases = [writes[0], reads[0]];
	fn node<'a>(buffers: &'a [BufferBinding<'a>]) -> ComputeDispatch<'a> {
		ComputeDispatch {
			kernel: KernelId::CryptographyMerkleCopyU8,
			buffers,
			push_constants: &[],
			workgroups: [1, 1, 1],
		}
	}
	validate_split_recording_access(&[node(&writes), node(&reads)])?;
	for dispatches in [
		vec![node(&reads), node(&writes)],
		vec![node(&updates), node(&reads)],
		vec![node(&aliases)],
	] {
		assert_eq!(
			validate_split_recording_access(&dispatches)
				.unwrap_err()
				.kind(),
			ErrorKind::FailedPrecondition
		);
	}
	// Admission is transactional: even successful validation publishes no producer.
	assert_eq!(
		intermediate.try_read::<u8>().unwrap_err().kind(),
		ErrorKind::NotReady
	);
	intermediate.storage().mark_failed();
	assert_eq!(
		validate_split_recording_access(&[node(&writes), node(&reads)])
			.unwrap_err()
			.kind(),
		ErrorKind::FailedPrecondition
	);
	Ok(())
}
