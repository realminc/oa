use super::*;
use crate::runtime::device::secret_buffer::SecretBuffer;

pub(in crate::runtime) fn assert_mlkem_cache_preflight(
	device: &Arc<Device>,
	secrets: [&SecretBuffer; 4],
	public_buffers: [&Buffer; 3],
) -> Result<()> {
	let [seed, shared, private, foreign_secret] = secrets;
	let [public, ciphertext, status] = public_buffers;
	for candidate in [foreign_secret, seed] {
		assert!(matches!(SecretConsumerOperands::encaps(
			device, [seed.binding()?, candidate.binding()?],
			public, ciphertext, status, 2), Err(error)
			if error.kind() == crate::ErrorKind::InvalidArgument));
		assert!(matches!(SecretConsumerOperands::decaps(
			device, [private.binding()?, candidate.binding()?],
			ciphertext, 2), Err(error)
			if error.kind() == crate::ErrorKind::InvalidArgument));
	}
	let _ = shared;
	Ok(())
}
