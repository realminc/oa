//! Extension compute kernels for SDK and third-party workloads.
//!
//! [`Kernel`] compiles caller-owned SPIR-V into a device compute pipeline and
//! [`Engine::dispatch_kernel`] records it through the same eager batching,
//! barrier, and event machinery as built-in operations. This is the supported
//! way to extend `oa` with custom shaders without forking the closed in-lib
//! kernel registry; the `oa_sdk` crate builds its workload kernels on it.
//!
//! ## Kernel ABI
//!
//! Extension kernels follow the OA compute ABI: the first push-constant words
//! are buffer descriptor indices, one per declared binding in binding order,
//! followed by the caller's `u32`/`f32` push words. Buffer bindings are
//! positional [`KernelBinding`] entries. The entry point is `main` and the
//! workgroup size is declared at compile time.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::runtime::{BufferBinding, Device, PushConstant};

use crate::{Engine, Error, Matrix, Result};

/// A compiled extension compute kernel on one engine's device.
///
/// Compile cost is paid once at construction; identical SPIR-V on the same
/// device reuses the cached pipeline. The kernel may be dispatched any number
/// of times via [`Engine::dispatch_kernel`]. It is not part of the semantic
/// graph or fusion system: dispatches record as physical nodes with the
/// caller-declared binding accesses.
pub struct Kernel {
	name: &'static str,
	key: Vec<u8>,
	bindings: usize,
	push_words: usize,
	device: Arc<Device>,
}

/// One storage binding in an extension-kernel dispatch.
///
/// The declared access drives barrier planning: reads may overlap, writes
/// serialize against the storage's other recorded work. Declaring `Write`
/// on a binding the shader also reads is a contract violation; use
/// [`KernelBinding::ReadWrite`].
#[derive(Clone, Copy)]
pub enum KernelBinding<'a> {
	/// Read-only storage binding.
	Read(&'a Matrix),
	/// Write-only storage binding.
	Write(&'a Matrix),
	/// Read-modify-write storage binding.
	ReadWrite(&'a Matrix),
}

/// One four-byte push-constant word appended after the binding indices.
#[derive(Clone, Copy, Debug)]
pub enum KernelPush {
	/// Unsigned 32-bit word.
	U32(u32),
	/// IEEE-754 32-bit word, bit-cast in place.
	F32(f32),
}

impl Kernel {
	/// Compile SPIR-V bytes into a dispatchable extension kernel.
	///
	/// `spirv` is the little-endian SPIR-V byte stream, exactly as produced by
	/// `slangc`/`glslc` or embedded via the `include_bytes!` macro.
	///
	/// `bindings` is the number of storage-buffer slots the kernel declares;
	/// those leading push-constant words are reserved for descriptor indices
	/// and supplied automatically at dispatch. The remaining reflected
	/// push-constant words are the caller's push surface — validated against
	/// the actual SPIR-V at compile time.
	///
	/// `workgroup_size` must match the shader's `local_size` declaration; it
	/// is validated against device limits here and again per dispatch.
	///
	/// # Errors
	///
	/// Returns an error when the bytes are not valid compute SPIR-V, when the
	/// reflected push-constant range is smaller than `bindings` words, when
	/// `workgroup_size` exceeds device limits, or when pipeline creation fails.
	pub fn compile(
		engine: &Engine,
		name: &'static str,
		spirv: &[u8],
		bindings: u32,
		workgroup_size: [u32; 3],
	) -> Result<Self> {
		let device = engine.handle().device();
		let words = crate::runtime::decode_spirv_bytes(spirv)?;
		let mut key = DefaultHasher::new();
		name.hash(&mut key);
		spirv.hash(&mut key);
		bindings.hash(&mut key);
		workgroup_size.hash(&mut key);
		let key = key.finish().to_le_bytes().to_vec();
		let pipeline = device.admit_extension_pipeline(key.clone(), &words, workgroup_size)?;
		let push_words = usize::try_from(pipeline.push_constant_size()).map_err(|_| {
			Error::internal(format!(
				"extension kernel {name} push-constant size does not fit this platform"
			))
		})? / 4;
		let push_words = push_words.checked_sub(bindings as usize).ok_or_else(|| {
			Error::invalid_argument(format!(
				"extension kernel {name} declares {bindings} buffer bindings but its \
				 push-constant block holds fewer words"
			))
		})?;
		Ok(Self {
			name,
			key,
			bindings: bindings as usize,
			push_words,
			device,
		})
	}

	/// The kernel's caller push-constant surface in four-byte words.
	///
	/// [`Engine::dispatch_kernel`] requires exactly this many
	/// [`KernelPush`] entries.
	pub const fn push_words(&self) -> usize {
		self.push_words
	}

	/// The kernel's declared storage-binding count.
	pub const fn binding_count(&self) -> usize {
		self.bindings
	}

	pub(crate) const fn name(&self) -> &'static str {
		self.name
	}

	pub(crate) fn key(&self) -> &[u8] {
		&self.key
	}
}

impl Engine {
	/// Record one eager extension-kernel dispatch.
	///
	/// The dispatch joins the engine's pending batch like any eager operation:
	/// barriers against other recorded work are derived from the declared
	/// binding accesses, and completion is observed through
	/// [`Engine::checkpoint`] or any host read.
	///
	/// # Errors
	///
	/// Returns `InvalidArgument` when `kernel` belongs to another engine, when
	/// `bindings`/`push_constants` counts do not match the kernel's declared
	/// ABI, or when `workgroups` is zero or exceeds device limits; returns the
	/// recording errors from the underlying session otherwise.
	pub fn dispatch_kernel(
		&self,
		kernel: &Kernel,
		bindings: &[KernelBinding<'_>],
		push_constants: &[KernelPush],
		workgroups: [u32; 3],
	) -> Result<()> {
		if bindings.len() != kernel.bindings {
			return Err(Error::invalid_argument(format!(
				"extension kernel {} declares {} bindings, got {}",
				kernel.name,
				kernel.bindings,
				bindings.len()
			)));
		}
		if push_constants.len() != kernel.push_words {
			return Err(Error::invalid_argument(format!(
				"extension kernel {} declares {} push words, got {}",
				kernel.name,
				kernel.push_words,
				push_constants.len()
			)));
		}
		let resolved = bindings
			.iter()
			.map(|binding| match *binding {
				KernelBinding::Read(matrix) => BufferBinding::read(matrix.storage()),
				KernelBinding::Write(matrix) => BufferBinding::write(matrix.storage()),
				KernelBinding::ReadWrite(matrix) => BufferBinding::read_write(matrix.storage()),
			})
			.collect::<Vec<_>>();
		let pushes = push_constants
			.iter()
			.map(|push| match *push {
				KernelPush::U32(value) => PushConstant::U32(value),
				KernelPush::F32(value) => PushConstant::F32(value),
			})
			.collect::<Vec<_>>();
		if !std::sync::Arc::ptr_eq(&kernel.device, &self.handle().device()) {
			return Err(Error::invalid_argument(format!(
				"extension kernel {} was compiled for a different engine",
				kernel.name
			)));
		}
		self
			.handle()
			.record_extension(kernel, &resolved, &pushes, workgroups)
	}
}

#[cfg(test)]
#[path = "../../../test/rs/runtime/compute_unit.rs"]
mod tests;
