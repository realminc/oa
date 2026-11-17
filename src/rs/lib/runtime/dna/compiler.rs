//! Dynamic compilation services and owned, validated SPIR-V artifacts.
//! Engine owns service lifetime; device code publishes Vulkan pipelines.

mod artifact;
mod elementwise;
mod slang;
mod spirv;
mod tile;
pub(in crate::runtime) mod worker;

use crate::Error;
use crate::runtime::dna::ir::elementwise::Program;
pub(in crate::runtime) use artifact::JitArtifacts;
pub(in crate::runtime) use slang::SlangCompiler;
use spirv::{validate_program_abi, validate_with_spirv_tools};

pub(in crate::runtime) struct CompiledChain {
	words: Vec<u32>,
	/// Emitted-code identity; empty for the manual Phase 1 recipe artifact.
	pub(in crate::runtime) key: Vec<u8>,
	/// Storage-buffer binding count baked into the descriptor ABI.
	pub(in crate::runtime) bindings: u32,
	pub(in crate::runtime) compile_time: std::time::Duration,
	#[cfg_attr(
		not(test),
		expect(
			dead_code,
			reason = "route observations report these only when adaptive routing is admitted"
		)
	)]
	pub(in crate::runtime) validation_time: std::time::Duration,
	pub(in crate::runtime) bounded: bool,
}

impl CompiledChain {
	pub(in crate::runtime) fn words(&self) -> &[u32] {
		&self.words
	}
}

fn jit_error(message: impl std::fmt::Display) -> Error {
	Error::backend_failure(
		"Slang",
		"runtime specialization",
		std::io::Error::other(message.to_string()),
	)
}

#[cfg(test)]
#[path = "../../../../../test/rs/runtime/jit_compiler_unit.rs"]
pub(in crate::runtime) mod tests;
