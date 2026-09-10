use crate::{Audio, Error, Image, Matrix, OpAttribute, OperationContract, Result};

use super::{Storage, shader::KernelId};

/// Borrowed semantic inputs and outputs for one executable lowering.
pub(crate) struct SemanticDispatch<'a> {
	pub(crate) contract: OperationContract,
	pub(crate) inputs: &'a [&'a Matrix],
	pub(crate) outputs: &'a [&'a Matrix],
	pub(crate) attributes: &'a [OpAttribute],
}

/// Borrowed semantic values for a contract with explicitly absent inputs.
pub(crate) struct OptionalSemanticDispatch<'a> {
	pub(crate) contract: OperationContract,
	pub(crate) inputs: &'a [Option<&'a Matrix>],
	pub(crate) outputs: &'a [&'a Matrix],
	pub(crate) attributes: &'a [OpAttribute],
}

impl OptionalSemanticDispatch<'_> {
	pub(crate) fn validate_kernel(&self, kernel: KernelId) -> Result<()> {
		let registered = kernel.semantic_contract().ok_or_else(|| {
			Error::internal(format!(
				"lowering-only kernel {} cannot record a direct semantic operation",
				kernel.report_name()
			))
		})?;
		if registered.name() != self.contract.name() || registered.hash() != self.contract.hash() {
			return Err(Error::internal(format!(
				"kernel {} is registered for {}, not {}",
				kernel.report_name(),
				registered.name(),
				self.contract.name()
			)));
		}
		Ok(())
	}
}

impl SemanticDispatch<'_> {
	pub(crate) fn validate_kernel(&self, kernel: KernelId) -> Result<()> {
		let registered = kernel.semantic_contract().ok_or_else(|| {
			Error::internal(format!(
				"lowering-only kernel {} cannot record a direct semantic operation",
				kernel.report_name()
			))
		})?;
		if registered.name() != self.contract.name() || registered.hash() != self.contract.hash() {
			return Err(Error::internal(format!(
				"kernel {} is registered for {}, not {}",
				kernel.report_name(),
				registered.name(),
				self.contract.name()
			)));
		}
		Ok(())
	}
}

/// Semantic output whose public value kind can differ from its Audio input.
pub(crate) enum AudioSemanticOutput<'a> {
	Audio(&'a Audio),
	Matrix(&'a Matrix),
}

/// Borrowed Audio-valued semantic operation for one or more executable nodes.
pub(crate) struct AudioSemanticDispatch<'a> {
	pub(crate) contract: OperationContract,
	pub(crate) inputs: &'a [&'a Audio],
	pub(crate) outputs: &'a [AudioSemanticOutput<'a>],
	pub(crate) attributes: &'a [OpAttribute],
}

impl AudioSemanticDispatch<'_> {
	pub(crate) fn validate_kernel(&self, kernel: KernelId) -> Result<()> {
		let registered = kernel.semantic_contract().ok_or_else(|| {
			Error::internal(format!(
				"lowering-only kernel {} cannot record a direct Audio operation",
				kernel.report_name()
			))
		})?;
		if registered.name() != self.contract.name() || registered.hash() != self.contract.hash() {
			return Err(Error::internal(format!(
				"kernel {} is registered for {}, not {}",
				kernel.report_name(),
				registered.name(),
				self.contract.name()
			)));
		}
		Ok(())
	}
}

/// Semantic input accepted by an Image operation.
///
/// Spatial transforms keep pixels as an [`Image`] while maps and transform
/// coefficients remain ordinary [`Matrix`] values in the semantic graph.
pub(crate) enum ImageSemanticInput<'a> {
	Image(&'a Image),
	Matrix(&'a Matrix),
}

/// Borrowed Image-valued semantic operation for one executable lowering.
pub(crate) struct ImageSemanticDispatch<'a> {
	pub(crate) contract: OperationContract,
	pub(crate) inputs: &'a [ImageSemanticInput<'a>],
	pub(crate) outputs: &'a [&'a Image],
	pub(crate) attributes: &'a [OpAttribute],
}

impl ImageSemanticDispatch<'_> {
	pub(crate) fn validate_kernel(&self, kernel: KernelId) -> Result<()> {
		let registered = kernel.semantic_contract().ok_or_else(|| {
			Error::internal(format!(
				"lowering-only kernel {} cannot record a direct Image operation",
				kernel.report_name()
			))
		})?;
		if registered.name() != self.contract.name() || registered.hash() != self.contract.hash() {
			return Err(Error::internal(format!(
				"kernel {} is registered for {}, not {}",
				kernel.report_name(),
				registered.name(),
				self.contract.name()
			)));
		}
		Ok(())
	}
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
	pub(crate) kernel: KernelId,
	pub(crate) buffers: &'a [BufferBinding<'a>],
	pub(crate) push_constants: &'a [PushConstant],
	pub(crate) workgroups: [u32; 3],
}

#[cfg(test)]
mod tests {
	use super::{KernelId, SemanticDispatch};

	#[test]
	fn semantic_dispatch_rejects_a_kernel_from_another_operation() {
		let dispatch = SemanticDispatch {
			contract: crate::core::operation::matrix::SUB,
			inputs: &[],
			outputs: &[],
			attributes: &[],
		};
		let error = dispatch
			.validate_kernel(KernelId::MatrixAddF32)
			.expect_err("mismatched kernel was accepted");
		assert_eq!(error.kind(), crate::ErrorKind::Internal);
		assert!(error.message().contains("registered for oa::matrix::add"));
	}

	#[test]
	fn semantic_dispatch_rejects_a_lowering_only_kernel() {
		let dispatch = SemanticDispatch {
			contract: crate::core::operation::ml::LINEAR,
			inputs: &[],
			outputs: &[],
			attributes: &[],
		};
		let error = dispatch
			.validate_kernel(KernelId::MlQkvProjectionBiasF32)
			.expect_err("lowering-only kernel was accepted as a semantic operation");
		assert_eq!(error.kind(), crate::ErrorKind::Internal);
		assert!(error.message().contains("lowering-only kernel"));
	}
}
