use super::{Storage, shader::KernelId};

/// Declared access made by one executable buffer binding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BufferAccess {
	Read,
	Write,
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
}

/// One four-byte value in the reflected compute push-constant ABI.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum PushConstant {
	/// Descriptor index of the buffer at this binding position.
	StorageBuffer(usize),
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
