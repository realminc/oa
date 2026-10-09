use crate::{Audio, Error, Image, Matrix, OpAttribute, OperationContract, Result};

use super::shader::KernelId;

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
