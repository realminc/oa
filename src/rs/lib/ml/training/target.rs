//! Exact target-network synchronization shared by off-policy trainers.

use crate::{Engine, Error, Result, matrix};

use crate::ml::Module;

pub(super) fn sync_exact(
	engine: &Engine,
	pairs: &[(&dyn Module, &dyn Module)],
	algorithm: &'static str,
) -> Result<()> {
	let mut routes = Vec::new();
	for (source_module, destination_module) in pairs {
		let source = source_module.all_named_parameters()?;
		let destination = destination_module.all_named_parameters()?;
		if source.len() != destination.len() {
			return Err(Error::invalid_argument(format!(
				"{algorithm} source and target modules have different parameter counts"
			)));
		}
		for (source, destination) in source.iter().zip(&destination) {
			let source_parameter = source.parameter();
			let destination_parameter = destination.parameter();
			let source_data = source_parameter.data();
			let destination_data = destination_parameter.data();
			if source.path() != destination.path()
				|| source_parameter.same_as(&destination_parameter)
				|| source_data.shape() != destination_data.shape()
				|| source_data.dtype() != destination_data.dtype()
				|| !engine.owns_matrix(&source_data)
				|| !engine.owns_matrix(&destination_data)
			{
				return Err(Error::invalid_argument(format!(
					"{algorithm} source and target module schemas do not match"
				)));
			}
			destination_parameter.validate_can_update()?;
			routes.push((source_data, destination_parameter));
		}
	}
	let copies = routes
		.iter()
		.map(|(source, _)| matrix::copy(source))
		.collect::<Result<Vec<_>>>()?;
	for ((_, destination), copied) in routes.into_iter().zip(copies) {
		destination.replace_data(copied)?;
		destination.set_requires_grad(false);
	}
	engine.checkpoint()?.wait()
}
