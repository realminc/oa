//! Shared optional-bias storage policy for the RNN and GRU providers.
//!
//! This module owns no operation or adjoint. Absent biases receive producer-only
//! storage; provider push constants determine whether a kernel reads that storage.

use crate::{Error, Matrix, Result};

pub(super) const MAX_HIDDEN_SIZE: usize = 1024;

pub(super) fn optional_bias(
	weight_hh: &Matrix,
	bias_hh: Option<&Matrix>,
	operation: &'static str,
) -> Result<Matrix> {
	match bias_hh {
		Some(value) => Ok(value.clone()),
		None => {
			let [gate_count, _] = weight_hh.shape() else {
				return Err(Error::invalid_argument(format!(
					"{operation} recurrent weight must have shape [3H, H]"
				)));
			};
			Matrix::allocate(
				weight_hh.engine_handle(),
				vec![*gate_count],
				*gate_count,
				crate::DType::F32,
			)
		}
	}
}
