use super::*;
use crate::runtime::device::secret_buffer::SecretBuffer;

pub(in crate::runtime) fn assert_mlkem_cache_preflight(
	device: &Arc<Device>,
	secrets: [&SecretBuffer; 6],
	public_buffers: [&Buffer; 3],
) -> Result<()> {
	let [seed, shared, private, workspace, short, foreign_workspace] = secrets;
	let [public, ciphertext, status] = public_buffers;
	for candidate in [short, foreign_workspace, seed] {
		assert!(matches!(SecretConsumerOperands::encaps(
			device, [seed.binding()?, shared.binding()?, candidate.binding()?],
			public, ciphertext, status, 2), Err(error)
			if error.kind() == crate::ErrorKind::InvalidArgument));
		assert!(matches!(SecretConsumerOperands::decaps(
			device, [private.binding()?, shared.binding()?, candidate.binding()?],
			ciphertext, 2), Err(error)
			if error.kind() == crate::ErrorKind::InvalidArgument));
	}
	assert!(matches!(SecretConsumerOperands::encaps(
		device, [seed.binding()?, seed.binding()?, workspace.binding()?],
		public, ciphertext, status, 2), Err(error)
		if error.kind() == crate::ErrorKind::InvalidArgument));
	Ok(())
}
