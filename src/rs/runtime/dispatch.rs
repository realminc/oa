use crate::{Matrix, OpAttribute, OperationContract};

use super::{Storage, shader::KernelId};

/// Borrowed semantic inputs and outputs for one executable lowering.
pub(crate) struct SemanticDispatch<'a> {
	pub(crate) contract: OperationContract,
	pub(crate) inputs: &'a [&'a Matrix],
	pub(crate) outputs: &'a [&'a Matrix],
	pub(crate) attributes: &'a [OpAttribute],
}

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

/// Backend-neutral executable compute description.
///
/// Domain lowering creates this value. The engine submits it generically; it
/// never grows operation-specific entry points.
pub(crate) struct ComputeDispatch<'a> {
	pub(crate) operation: &'static str,
	pub(crate) kernel: KernelId,
	pub(crate) buffers: &'a [BufferBinding<'a>],
	pub(crate) push_constants: &'a [PushConstant],
	pub(crate) workgroups: [u32; 3],
}
