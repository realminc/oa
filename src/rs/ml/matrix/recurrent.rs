//! ML-owned Matrix recurrent operations.

use crate::{Matrix, Result};

use crate::ml::{Parameter, autograd, lowering::recurrent as dispatch};

/// Apply one Elman RNN cell update to a caller-supplied hidden state.
///
/// `gates_i` and `hidden` have shape `[batch, hidden]`, `weight_hh` has
/// `[hidden, hidden]`, and an optional recurrent bias has `[hidden]`.
///
/// # Errors
///
/// Returns an error unless all values are nonempty same-engine F32 matrices,
/// shapes agree, hidden size is at most 1024, or runtime recording fails.
pub fn rnn_cell(
	gates_i: &Matrix,
	hidden: &Matrix,
	weight_hh: &Matrix,
	bias_hh: Option<&Matrix>,
) -> Result<Matrix> {
	let bias_value = optional_bias(weight_hh, bias_hh, "oa::ml::matrix::rnn_cell")?;
	let has_bias = bias_hh.is_some();
	let result = dispatch::rnn_cell(gates_i, hidden, weight_hh, &bias_value, has_bias)?;
	autograd::record_rnn_cell(
		gates_i,
		hidden,
		&result.output,
		result.gates_h,
		None,
		weight_hh.clone(),
		None,
		bias_value,
		has_bias,
	)?;
	Ok(result.output)
}

#[allow(
	clippy::too_many_arguments,
	reason = "module forwarding preserves optional parameter identity separately from Matrix values"
)]
pub(in crate::ml) fn rnn_cell_parameterized(
	gates_i: &Matrix,
	hidden: &Matrix,
	weight: (Parameter, Matrix, u64),
	bias: Option<(Parameter, u64)>,
	bias_value: &Matrix,
	has_bias: bool,
) -> Result<Matrix> {
	let (weight_parameter, weight_value, weight_version) = weight;
	let result = dispatch::rnn_cell(gates_i, hidden, &weight_value, bias_value, has_bias)?;
	autograd::record_rnn_cell(
		gates_i,
		hidden,
		&result.output,
		result.gates_h,
		Some((weight_parameter, weight_version)),
		weight_value,
		bias,
		bias_value.clone(),
		has_bias,
	)?;
	Ok(result.output)
}

pub(in crate::ml) fn rnn_cell_backward(
	gates_i: &Matrix,
	gates_h: &Matrix,
	hidden: &Matrix,
	output_gradient: &Matrix,
	weight_hh: &Matrix,
	bias_hh: &Matrix,
	has_bias: bool,
) -> Result<(Matrix, Matrix, Matrix, Matrix)> {
	dispatch::rnn_cell_backward(
		gates_i,
		gates_h,
		hidden,
		output_gradient,
		weight_hh,
		bias_hh,
		has_bias,
	)
}

/// Run a whole-sequence Elman recurrence from a zero hidden state.
///
/// `gates_i` has shape `[batch, sequence, hidden]`, `weight_hh` has
/// `[hidden, hidden]`, and an optional recurrent bias has `[hidden]`.
///
/// # Errors
///
/// Returns an error unless all values are nonempty same-engine F32 matrices,
/// shapes agree, hidden size is at most 1024, or runtime recording fails.
pub fn rnn_scan(gates_i: &Matrix, weight_hh: &Matrix, bias_hh: Option<&Matrix>) -> Result<Matrix> {
	let bias_value = optional_bias(weight_hh, bias_hh, "oa::ml::matrix::rnn_scan")?;
	let has_bias = bias_hh.is_some();
	let result = dispatch::rnn_scan(gates_i, weight_hh, &bias_value, has_bias)?;
	autograd::record_rnn_scan(
		gates_i,
		&result.output,
		result.hidden_previous,
		None,
		weight_hh.clone(),
		None,
		bias_value,
		has_bias,
	)?;
	Ok(result.output)
}

pub(in crate::ml) fn rnn_scan_parameterized(
	gates_i: &Matrix,
	weight: (Parameter, Matrix, u64),
	bias: Option<(Parameter, u64)>,
	bias_value: &Matrix,
	has_bias: bool,
) -> Result<Matrix> {
	let (weight_parameter, weight_value, weight_version) = weight;
	let result = dispatch::rnn_scan(gates_i, &weight_value, bias_value, has_bias)?;
	autograd::record_rnn_scan(
		gates_i,
		&result.output,
		result.hidden_previous,
		Some((weight_parameter, weight_version)),
		weight_value,
		bias,
		bias_value.clone(),
		has_bias,
	)?;
	Ok(result.output)
}

pub(in crate::ml) fn rnn_scan_backward(
	output_gradient: &Matrix,
	gates_i: &Matrix,
	hidden_previous: &Matrix,
	weight_hh: &Matrix,
	bias_hh: &Matrix,
	has_bias: bool,
) -> Result<(Matrix, Matrix, Matrix)> {
	dispatch::rnn_scan_backward(
		output_gradient,
		gates_i,
		hidden_previous,
		weight_hh,
		bias_hh,
		has_bias,
	)
}

/// Run a whole-sequence GRU recurrence from a zero hidden state.
///
/// `gates_i` has shape `[batch, sequence, 3 * hidden]`, `weight_hh` has
/// `[3 * hidden, hidden]`, and an optional `bias_hh` has `[3 * hidden]`. Gate
/// order is reset, update, candidate.
///
/// # Errors
///
/// Returns an error unless all values are nonempty same-engine F32 matrices,
/// shapes agree, hidden size is at most 1024, or runtime recording fails.
pub fn gru_scan(gates_i: &Matrix, weight_hh: &Matrix, bias_hh: Option<&Matrix>) -> Result<Matrix> {
	let bias_value = match bias_hh {
		Some(value) => value.clone(),
		None => {
			let [gate_count, _] = weight_hh.shape() else {
				return Err(crate::Error::invalid_argument(
					"oa::ml::matrix::gru_scan recurrent weight must have shape [3H, H]",
				));
			};
			Matrix::allocate(
				weight_hh.engine_handle(),
				vec![*gate_count],
				*gate_count,
				crate::DType::F32,
			)?
		}
	};
	let has_bias = bias_hh.is_some();
	let result = dispatch::gru_scan(gates_i, weight_hh, &bias_value, has_bias)?;
	autograd::record_gru_scan(
		gates_i,
		&result.output,
		result.hidden_previous,
		None,
		weight_hh.clone(),
		None,
		bias_value,
		has_bias,
	)?;
	Ok(result.output)
}

pub(in crate::ml) fn gru_scan_parameterized(
	gates_i: &Matrix,
	weight: (Parameter, Matrix, u64),
	bias: Option<(Parameter, u64)>,
	bias_value: &Matrix,
	has_bias: bool,
) -> Result<Matrix> {
	let (weight_parameter, weight_value, weight_version) = weight;
	let (bias_parameter, bias_version) = bias
		.map(|(parameter, version)| (Some(parameter), Some(version)))
		.unwrap_or((None, None));
	let result = dispatch::gru_scan(gates_i, &weight_value, bias_value, has_bias)?;
	autograd::record_gru_scan(
		gates_i,
		&result.output,
		result.hidden_previous,
		Some((weight_parameter, weight_version)),
		weight_value,
		bias_parameter.zip(bias_version),
		bias_value.clone(),
		has_bias,
	)?;
	Ok(result.output)
}

pub(in crate::ml) fn gru_scan_backward(
	output_gradient: &Matrix,
	gates_i: &Matrix,
	hidden_previous: &Matrix,
	weight_hh: &Matrix,
	bias_hh: &Matrix,
	has_bias: bool,
) -> Result<(Matrix, Matrix, Matrix)> {
	dispatch::gru_scan_backward(
		output_gradient,
		gates_i,
		hidden_previous,
		weight_hh,
		bias_hh,
		has_bias,
	)
}

/// Apply one GRU cell update to a caller-supplied hidden state.
///
/// `gates_i` has shape `[batch, 3 * hidden]`, `hidden` has `[batch, hidden]`,
/// `weight_hh` has `[3 * hidden, hidden]`, and an optional recurrent bias has
/// `[3 * hidden]`.
///
/// # Errors
///
/// Returns an error unless all values are nonempty same-engine F32 matrices,
/// shapes agree, hidden size is at most 1024, or runtime recording fails.
pub fn gru_cell(
	gates_i: &Matrix,
	hidden: &Matrix,
	weight_hh: &Matrix,
	bias_hh: Option<&Matrix>,
) -> Result<Matrix> {
	let bias_value = optional_bias(weight_hh, bias_hh, "oa::ml::matrix::gru_cell")?;
	let has_bias = bias_hh.is_some();
	let result = dispatch::gru_cell(gates_i, hidden, weight_hh, &bias_value, has_bias)?;
	autograd::record_gru_cell(
		gates_i,
		hidden,
		&result.output,
		result.gates_h,
		None,
		weight_hh.clone(),
		None,
		bias_value,
		has_bias,
	)?;
	Ok(result.output)
}

#[allow(
	clippy::too_many_arguments,
	reason = "module forwarding preserves optional parameter identity separately from Matrix values"
)]
pub(in crate::ml) fn gru_cell_parameterized(
	gates_i: &Matrix,
	hidden: &Matrix,
	weight: (Parameter, Matrix, u64),
	bias: Option<(Parameter, u64)>,
	bias_value: &Matrix,
	has_bias: bool,
) -> Result<Matrix> {
	let (weight_parameter, weight_value, weight_version) = weight;
	let result = dispatch::gru_cell(gates_i, hidden, &weight_value, bias_value, has_bias)?;
	autograd::record_gru_cell(
		gates_i,
		hidden,
		&result.output,
		result.gates_h,
		Some((weight_parameter, weight_version)),
		weight_value,
		bias,
		bias_value.clone(),
		has_bias,
	)?;
	Ok(result.output)
}

pub(in crate::ml) fn gru_cell_backward(
	gates_i: &Matrix,
	gates_h: &Matrix,
	hidden: &Matrix,
	output_gradient: &Matrix,
	weight_hh: &Matrix,
	bias_hh: &Matrix,
	has_bias: bool,
) -> Result<(Matrix, Matrix, Matrix, Matrix)> {
	dispatch::gru_cell_backward(
		gates_i,
		gates_h,
		hidden,
		output_gradient,
		weight_hh,
		bias_hh,
		has_bias,
	)
}

fn optional_bias(
	weight_hh: &Matrix,
	bias_hh: Option<&Matrix>,
	operation: &'static str,
) -> Result<Matrix> {
	match bias_hh {
		Some(value) => Ok(value.clone()),
		None => {
			let [gate_count, _] = weight_hh.shape() else {
				return Err(crate::Error::invalid_argument(format!(
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
