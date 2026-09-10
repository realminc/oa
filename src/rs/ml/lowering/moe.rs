//! Private lowering for sparse mixture-of-experts operations.

use crate::{
	DType, Error, Matrix, Result,
	runtime::{BufferBinding, KernelId, PushConstant},
};

use super::common::{record_semantic, shader_u32};

struct RouteGeometry {
	tokens: u32,
	experts: u32,
	routes_per_token: u32,
	probability_count: u32,
}

impl RouteGeometry {
	fn resolve(
		probabilities: &Matrix,
		expert_indices: &Matrix,
		operation: &'static str,
	) -> Result<Self> {
		let [tokens, experts] = probabilities.shape() else {
			return Err(Error::invalid_argument(format!(
				"{operation} probabilities must have shape [T, E]; found {:?}",
				probabilities.shape()
			)));
		};
		let [index_tokens, routes_per_token] = expert_indices.shape() else {
			return Err(Error::invalid_argument(format!(
				"{operation} expert indices must have shape [T, K]; found {:?}",
				expert_indices.shape()
			)));
		};
		if *experts == 0
			|| *routes_per_token == 0
			|| routes_per_token > experts
			|| tokens != index_tokens
			|| probabilities.dtype() != DType::F32
			|| expert_indices.dtype() != DType::I32
			|| !probabilities
				.engine_handle()
				.same_as(expert_indices.engine_handle())
		{
			return Err(Error::invalid_argument(format!(
				"{operation} requires same-engine F32 probabilities [T, E] and I32 indices [T, K] with 1 <= K <= E"
			)));
		}
		let probability_count = tokens.checked_mul(*experts).ok_or_else(|| {
			Error::invalid_argument(format!("{operation} probability size overflows usize"))
		})?;
		Ok(Self {
			tokens: shader_u32(*tokens, "token count", operation)?,
			experts: shader_u32(*experts, "expert count", operation)?,
			routes_per_token: shader_u32(*routes_per_token, "routes per token", operation)?,
			probability_count: shader_u32(
				probability_count,
				"probability element count",
				operation,
			)?,
		})
	}

	const fn push_constants(&self) -> [PushConstant; 3] {
		[
			PushConstant::U32(self.tokens),
			PushConstant::U32(self.experts),
			PushConstant::U32(self.routes_per_token),
		]
	}
}

pub(in crate::ml) fn route_weights(
	probabilities: &Matrix,
	expert_indices: &Matrix,
) -> Result<Matrix> {
	const OPERATION: &str = crate::core::operation::ml::MOE_ROUTE_WEIGHTS.name();
	let geometry = RouteGeometry::resolve(probabilities, expert_indices, OPERATION)?;
	let output = Matrix::allocate(
		probabilities.engine_handle(),
		expert_indices.shape().to_vec(),
		expert_indices.num_elements(),
		DType::F32,
	)?;
	if geometry.tokens != 0 {
		let buffers = [
			BufferBinding::read(probabilities.storage()),
			BufferBinding::read(expert_indices.storage()),
			BufferBinding::write(output.storage()),
		];
		let push_constants = geometry.push_constants();
		let kernel = KernelId::MlMoeRouteWeightsF32;
		record_semantic(
			&[probabilities, expert_indices],
			&[&output],
			&[],
			kernel,
			&buffers,
			&push_constants,
			kernel.linear_workgroups(geometry.tokens),
		)?;
	}
	Ok(output)
}

pub(in crate::ml) fn route_weights_backward(
	output_gradient: &Matrix,
	probabilities: &Matrix,
	expert_indices: &Matrix,
	route_weights: &Matrix,
) -> Result<Matrix> {
	const OPERATION: &str = crate::core::operation::ml::MOE_ROUTE_WEIGHTS_BACKWARD.name();
	let geometry = RouteGeometry::resolve(probabilities, expert_indices, OPERATION)?;
	if output_gradient.shape() != expert_indices.shape()
		|| route_weights.shape() != expert_indices.shape()
		|| output_gradient.dtype() != DType::F32
		|| route_weights.dtype() != DType::F32
		|| !probabilities
			.engine_handle()
			.same_as(output_gradient.engine_handle())
		|| !probabilities
			.engine_handle()
			.same_as(route_weights.engine_handle())
	{
		return Err(Error::invalid_argument(format!(
			"{OPERATION} requires same-engine F32 output gradients and route weights shaped like the I32 indices"
		)));
	}
	let probability_gradient = Matrix::allocate(
		probabilities.engine_handle(),
		probabilities.shape().to_vec(),
		probabilities.num_elements(),
		DType::F32,
	)?;
	if geometry.probability_count != 0 {
		let buffers = [
			BufferBinding::read(output_gradient.storage()),
			BufferBinding::read(probabilities.storage()),
			BufferBinding::read(expert_indices.storage()),
			BufferBinding::read(route_weights.storage()),
			BufferBinding::write(probability_gradient.storage()),
		];
		let push_constants = geometry.push_constants();
		let kernel = KernelId::MlMoeRouteWeightsBackwardF32;
		record_semantic(
			&[
				output_gradient,
				probabilities,
				expert_indices,
				route_weights,
			],
			&[&probability_gradient],
			&[],
			kernel,
			&buffers,
			&push_constants,
			kernel.linear_workgroups(geometry.probability_count),
		)?;
	}
	Ok(probability_gradient)
}
