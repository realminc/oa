//! Executable compute bindings produced by domain lowering.

use crate::runtime::{Storage, dna::kernel::KernelId};

/// Declared access made by one executable buffer binding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BufferAccess {
	Read,
	Write,
	ReadWrite,
}

/// One storage binding in an executable compute dispatch.
#[derive(Clone, Copy)]
pub(crate) struct BufferBinding<'a> {
	pub(crate) storage: &'a Storage,
	pub(crate) access: BufferAccess,
}

impl<'a> BufferBinding<'a> {
	pub(crate) const fn read(storage: &'a Storage) -> Self {
		Self {
			storage,
			access: BufferAccess::Read,
		}
	}

	pub(crate) const fn write(storage: &'a Storage) -> Self {
		Self {
			storage,
			access: BufferAccess::Write,
		}
	}

	pub(crate) const fn read_write(storage: &'a Storage) -> Self {
		Self {
			storage,
			access: BufferAccess::ReadWrite,
		}
	}
}

/// One four-byte value in the reflected compute push-constant ABI.
///
/// Buffer descriptor indices are runtime-owned ABI data and are prepended in
/// binding order when the dispatch is recorded.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum PushConstant {
	U32(u32),
	F32(f32),
}

/// Device-resident `VkDispatchIndirectCommand` source for one compute dispatch.
///
/// `storage` must contain a four-byte-aligned `offset` followed by three
/// initialized `u32` workgroup counts. A producer dispatch owns the argument
/// contents and proves the one-dimensional group bound; the host cannot
/// inspect GPU-authored dimensions at record time.
#[derive(Clone, Copy)]
pub(crate) struct IndirectDispatch<'a> {
	pub(crate) storage: &'a Storage,
	pub(crate) offset: u64,
}

/// Backend-neutral executable compute description.
///
/// Domain lowering creates this value. The engine submits it generically; it
/// never grows operation-specific entry points.
pub(crate) struct ComputeDispatch<'a> {
	pub(crate) kernel: KernelId,
	pub(crate) buffers: &'a [BufferBinding<'a>],
	pub(crate) push_constants: &'a [PushConstant],
	pub(crate) workgroups: [u32; 3],
	/// Optional GPU-authored launch dimensions. When present the recorded
	/// command is `vkCmdDispatchIndirect` and `workgroups` is unused.
	pub(crate) indirect: Option<IndirectDispatch<'a>>,
}
