//! Shared broadcast reduction for Matrix adjoints.

use crate::{Error, Matrix, Result, matrix as operations};

pub(crate) fn sum_to_shape(gradient: &Matrix, target_shape: &[usize]) -> Result<Matrix> {
	if gradient.shape() == target_shape {
		return Ok(gradient.clone());
	}
	if gradient.shape().len() < target_shape.len() {
		return Err(Error::internal(
			"broadcast adjoint target rank exceeds output-gradient rank",
		));
	}
	let gradient_shape = gradient.shape().to_vec();
	let leading = gradient_shape.len() - target_shape.len();
	let mut result = gradient.clone();
	for (axis, extent) in gradient_shape.iter().copied().enumerate() {
		let target_extent = axis
			.checked_sub(leading)
			.map_or(1, |target_axis| target_shape[target_axis]);
		if target_extent == 1 && extent > 1 {
			result = operations::sum(
				&result,
				i32::try_from(axis).map_err(|_| Error::internal("broadcast adjoint axis exceeds i32"))?,
			)?;
		}
	}
	if result.shape() != target_shape {
		result = result.reshape(target_shape.to_vec())?;
	}
	Ok(result)
}
