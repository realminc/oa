use super::*;

impl Device {
	pub(in crate::runtime) fn record_jit_chain(
		self: &Arc<Self>,
		chain: &crate::runtime::device::command::jit_lab::PreparedChain,
	) -> Result<super::super::command::ReusableCommandBuffer> {
		let mut pool = self
			.logical
			.command_pool
			.lock()
			.map_err(|_| Error::failed_precondition("compute command pool is poisoned"))?;
		chain.record(&mut pool, self)
	}
}

impl Device {
	/// Completed route observations `(signature, specialized samples, tape
	/// samples, selected)`, used by gated tests.
	pub(in crate::runtime) fn jit_observations(&self) -> Vec<(u64, usize, usize, bool)> {
		self
			.logical
			.jit
			.lock()
			.map(|mut fusion| fusion.observations())
			.unwrap_or_default()
	}
}
