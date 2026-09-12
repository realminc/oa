//! Private lowering for sparse mixture-of-experts operations.

use crate::{
	DType, Error, Matrix, Result,
	runtime::{BufferBinding, ComputeDispatch, KernelId, PushConstant, SemanticDispatch},
};

use super::common::{record_semantic, shader_u32};

pub(in crate::ml) struct CombineBackward {
	pub(in crate::ml) packed: Matrix,
	pub(in crate::ml) route_gate: Matrix,
}

pub(in crate::ml) struct GroupedLinearBackward {
	pub(in crate::ml) input: Matrix,
	pub(in crate::ml) weight: Matrix,
	pub(in crate::ml) bias: Matrix,
}

pub(in crate::ml) struct GroupedGemmBackward {
	pub(in crate::ml) input: Matrix,
	pub(in crate::ml) weight: Matrix,
}

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

pub(in crate::ml) fn gather(
	input: &Matrix,
	packed_token: &Matrix,
	inverse: &Matrix,
) -> Result<Matrix> {
	const OPERATION: &str = crate::core::operation::ml::MOE_GATHER.name();
	let [tokens, row_width] = input.shape() else {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} input must have shape [T, D]"
		)));
	};
	let [routes] = packed_token.shape() else {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} packed-token map must have shape [R]"
		)));
	};
	if input.dtype() != DType::F32
		|| packed_token.dtype() != DType::U32
		|| inverse.dtype() != DType::U32
		|| inverse.shape() != packed_token.shape()
		|| *tokens == 0
		|| *row_width == 0
		|| *routes == 0
		|| routes % tokens != 0
		|| !input.engine_handle().same_as(packed_token.engine_handle())
		|| !input.engine_handle().same_as(inverse.engine_handle())
	{
		return Err(Error::invalid_argument(format!(
			"{OPERATION} requires same-engine F32 input [T, D] and U32 packed-token/inverse maps [R] with R % T == 0"
		)));
	}
	let output_count = routes.checked_mul(*row_width).ok_or_else(|| {
		Error::invalid_argument(format!("{OPERATION} output size overflows usize"))
	})?;
	let routes_u32 = shader_u32(*routes, "route count", OPERATION)?;
	let row_width_u32 = shader_u32(*row_width, "row width", OPERATION)?;
	let output_count_u32 = shader_u32(output_count, "output element count", OPERATION)?;
	let output = Matrix::allocate(
		input.engine_handle(),
		vec![*routes, *row_width],
		output_count,
		DType::F32,
	)?;
	let buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::read(packed_token.storage()),
		BufferBinding::write(output.storage()),
	];
	let push_constants = [
		PushConstant::U32(routes_u32),
		PushConstant::U32(row_width_u32),
	];
	let kernel = KernelId::MlMoeGatherF32;
	record_semantic(
		&[input, packed_token, inverse],
		&[&output],
		&[],
		kernel,
		&buffers,
		&push_constants,
		kernel.linear_workgroups(output_count_u32),
	)?;
	Ok(output)
}

pub(in crate::ml) fn gather_backward(
	source: &Matrix,
	inverse: &Matrix,
	output_rows: usize,
) -> Result<Matrix> {
	const OPERATION: &str = crate::core::operation::ml::MOE_GATHER_BACKWARD.name();
	let [routes, row_width] = source.shape() else {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} source must have shape [R, D]"
		)));
	};
	if source.dtype() != DType::F32
		|| inverse.dtype() != DType::U32
		|| inverse.shape() != [*routes]
		|| *routes == 0
		|| *row_width == 0
		|| output_rows == 0
		|| routes % output_rows != 0
		|| !source.engine_handle().same_as(inverse.engine_handle())
	{
		return Err(Error::invalid_argument(format!(
			"{OPERATION} requires same-engine F32 source [R, D], U32 inverse [R], and positive output rows dividing R"
		)));
	}
	let output_count = output_rows.checked_mul(*row_width).ok_or_else(|| {
		Error::invalid_argument(format!("{OPERATION} output size overflows usize"))
	})?;
	let routes_u32 = shader_u32(*routes, "route count", OPERATION)?;
	let row_width_u32 = shader_u32(*row_width, "row width", OPERATION)?;
	let tokens_u32 = shader_u32(output_rows, "output row count", OPERATION)?;
	let routes_per_token_u32 = shader_u32(routes / output_rows, "routes per token", OPERATION)?;
	let output_count_u32 = shader_u32(output_count, "output element count", OPERATION)?;
	let output = Matrix::allocate(
		source.engine_handle(),
		vec![output_rows, *row_width],
		output_count,
		DType::F32,
	)?;
	let buffers = [
		BufferBinding::read(source.storage()),
		BufferBinding::read(inverse.storage()),
		BufferBinding::write(output.storage()),
	];
	let push_constants = [
		PushConstant::U32(routes_u32),
		PushConstant::U32(row_width_u32),
		PushConstant::U32(tokens_u32),
		PushConstant::U32(routes_per_token_u32),
	];
	let attributes = [crate::OpAttribute::SignedInteger {
		name: "output_rows".into(),
		value: i64::try_from(output_rows)
			.map_err(|_| Error::invalid_argument(format!("{OPERATION} output rows exceed i64")))?,
	}];
	let kernel = KernelId::MlMoeGatherBackwardF32;
	record_semantic(
		&[source, inverse],
		&[&output],
		&attributes,
		kernel,
		&buffers,
		&push_constants,
		kernel.linear_workgroups(output_count_u32),
	)?;
	Ok(output)
}

pub(in crate::ml) fn combine(
	packed: &Matrix,
	route_gate: &Matrix,
	inverse: &Matrix,
	packed_slot: &Matrix,
) -> Result<Matrix> {
	let geometry = CombineGeometry::resolve(
		packed,
		route_gate,
		inverse,
		packed_slot,
		crate::core::operation::ml::MOE_COMBINE.name(),
	)?;
	let output = Matrix::allocate(
		packed.engine_handle(),
		vec![geometry.tokens as usize, geometry.row_width as usize],
		geometry.output_count as usize,
		DType::F32,
	)?;
	let buffers = [
		BufferBinding::read(packed.storage()),
		BufferBinding::read(route_gate.storage()),
		BufferBinding::read(inverse.storage()),
		BufferBinding::write(output.storage()),
	];
	let push_constants = geometry.push_constants();
	let kernel = KernelId::MlMoeCombineF32;
	record_semantic(
		&[packed, route_gate, inverse, packed_slot],
		&[&output],
		&[],
		kernel,
		&buffers,
		&push_constants,
		kernel.linear_workgroups(geometry.output_count),
	)?;
	Ok(output)
}

pub(in crate::ml) fn combine_backward(
	output_gradient: &Matrix,
	packed: &Matrix,
	route_gate: &Matrix,
	inverse: &Matrix,
	packed_slot: &Matrix,
) -> Result<CombineBackward> {
	const OPERATION: &str = crate::core::operation::ml::MOE_COMBINE_BACKWARD.name();
	let geometry = CombineGeometry::resolve(packed, route_gate, inverse, packed_slot, OPERATION)?;
	if output_gradient.dtype() != DType::F32
		|| output_gradient.shape() != [geometry.tokens as usize, geometry.row_width as usize]
		|| !packed
			.engine_handle()
			.same_as(output_gradient.engine_handle())
	{
		return Err(Error::invalid_argument(format!(
			"{OPERATION} output gradient must be same-engine F32 [T, D]"
		)));
	}
	let packed_gradient = Matrix::allocate(
		packed.engine_handle(),
		packed.shape().to_vec(),
		packed.num_elements(),
		DType::F32,
	)?;
	let route_gate_gradient = Matrix::allocate(
		packed.engine_handle(),
		route_gate.shape().to_vec(),
		route_gate.num_elements(),
		DType::F32,
	)?;
	let buffers = [
		BufferBinding::read(output_gradient.storage()),
		BufferBinding::read(packed.storage()),
		BufferBinding::read(route_gate.storage()),
		BufferBinding::read(inverse.storage()),
		BufferBinding::read(packed_slot.storage()),
		BufferBinding::write(packed_gradient.storage()),
		BufferBinding::write(route_gate_gradient.storage()),
	];
	let push_constants = geometry.push_constants();
	let dispatch_count = geometry
		.routes
		.checked_mul(geometry.row_width)
		.ok_or_else(|| {
			Error::invalid_argument(format!("{OPERATION} dispatch count exceeds u32"))
		})?;
	let kernel = KernelId::MlMoeCombineBackwardF32;
	record_semantic(
		&[output_gradient, packed, route_gate, inverse, packed_slot],
		&[&packed_gradient, &route_gate_gradient],
		&[],
		kernel,
		&buffers,
		&push_constants,
		kernel.linear_workgroups(dispatch_count),
	)?;
	Ok(CombineBackward {
		packed: packed_gradient,
		route_gate: route_gate_gradient,
	})
}

struct CombineGeometry {
	tokens: u32,
	routes_per_token: u32,
	row_width: u32,
	routes: u32,
	output_count: u32,
}

impl CombineGeometry {
	fn resolve(
		packed: &Matrix,
		route_gate: &Matrix,
		inverse: &Matrix,
		packed_slot: &Matrix,
		operation: &'static str,
	) -> Result<Self> {
		let [routes, row_width] = packed.shape() else {
			return Err(Error::invalid_argument(format!(
				"{operation} packed input must have shape [R, D]"
			)));
		};
		let [tokens, routes_per_token] = route_gate.shape() else {
			return Err(Error::invalid_argument(format!(
				"{operation} route gates must have shape [T, K]"
			)));
		};
		let expected_routes = tokens.checked_mul(*routes_per_token).ok_or_else(|| {
			Error::invalid_argument(format!("{operation} route count overflows usize"))
		})?;
		if packed.dtype() != DType::F32
			|| route_gate.dtype() != DType::F32
			|| inverse.dtype() != DType::U32
			|| packed_slot.dtype() != DType::U32
			|| inverse.shape() != [*routes]
			|| packed_slot.shape() != [*routes]
			|| *routes == 0
			|| *row_width == 0
			|| *tokens == 0
			|| *routes_per_token == 0
			|| *routes != expected_routes
			|| !packed.engine_handle().same_as(route_gate.engine_handle())
			|| !packed.engine_handle().same_as(inverse.engine_handle())
			|| !packed.engine_handle().same_as(packed_slot.engine_handle())
		{
			return Err(Error::invalid_argument(format!(
				"{operation} requires same-engine F32 packed [T*K, D], F32 route gates [T, K], and U32 inverse/packed-slot maps [T*K]"
			)));
		}
		let output_count = tokens.checked_mul(*row_width).ok_or_else(|| {
			Error::invalid_argument(format!("{operation} output size overflows usize"))
		})?;
		Ok(Self {
			tokens: shader_u32(*tokens, "token count", operation)?,
			routes_per_token: shader_u32(*routes_per_token, "routes per token", operation)?,
			row_width: shader_u32(*row_width, "row width", operation)?,
			routes: shader_u32(*routes, "route count", operation)?,
			output_count: shader_u32(output_count, "output element count", operation)?,
		})
	}

	const fn push_constants(&self) -> [PushConstant; 4] {
		[
			PushConstant::U32(self.tokens),
			PushConstant::U32(self.routes_per_token),
			PushConstant::U32(self.row_width),
			PushConstant::U32(self.routes),
		]
	}
}

struct GroupedLinearGeometry {
	rows: u32,
	output_features: u32,
	input_features: u32,
	experts: u32,
}

impl GroupedLinearGeometry {
	fn resolve(
		input: &Matrix,
		weight: &Matrix,
		bias: Option<&Matrix>,
		offsets: &Matrix,
		operation: &'static str,
	) -> Result<Self> {
		let [rows, input_features] = input.shape() else {
			return Err(Error::invalid_argument(format!(
				"{operation} input must have shape [R, K]"
			)));
		};
		let [experts, output_features, weight_input_features] = weight.shape() else {
			return Err(Error::invalid_argument(format!(
				"{operation} weight must have shape [E, N, K]"
			)));
		};
		let offset_count = experts.checked_add(1).ok_or_else(|| {
			Error::invalid_argument(format!("{operation} expert count overflows usize"))
		})?;
		if *rows == 0
			|| *input_features == 0
			|| *output_features == 0
			|| *experts == 0
			|| input_features != weight_input_features
			|| offsets.shape() != [offset_count]
			|| input.dtype() != DType::F32
			|| weight.dtype() != DType::F32
			|| offsets.dtype() != DType::U32
			|| !input.engine_handle().same_as(weight.engine_handle())
			|| !input.engine_handle().same_as(offsets.engine_handle())
			|| bias.is_some_and(|bias| {
				bias.shape() != [*experts, *output_features]
					|| bias.dtype() != DType::F32
					|| !input.engine_handle().same_as(bias.engine_handle())
			}) {
			return Err(Error::invalid_argument(format!(
				"{operation} requires same-engine nonempty F32 input [R, K], weight [E, N, K], bias [E, N], and U32 expert offsets [E+1]"
			)));
		}
		Ok(Self {
			rows: shader_u32(*rows, "row count", operation)?,
			output_features: shader_u32(*output_features, "output feature count", operation)?,
			input_features: shader_u32(*input_features, "input feature count", operation)?,
			experts: shader_u32(*experts, "expert count", operation)?,
		})
	}

	const fn push_constants(&self) -> [PushConstant; 4] {
		[
			PushConstant::U32(self.rows),
			PushConstant::U32(self.output_features),
			PushConstant::U32(self.input_features),
			PushConstant::U32(self.experts),
		]
	}
}

pub(in crate::ml) fn grouped_gemm_m(
	input: &Matrix,
	weight: &Matrix,
	offsets: &Matrix,
) -> Result<Matrix> {
	const OPERATION: &str = crate::core::operation::ml::GROUPED_GEMM_M.name();
	let geometry = GroupedLinearGeometry::resolve(input, weight, None, offsets, OPERATION)?;
	let output_count = (geometry.rows as usize)
		.checked_mul(geometry.output_features as usize)
		.ok_or_else(|| {
			Error::invalid_argument(format!("{OPERATION} output size overflows usize"))
		})?;
	let output = Matrix::allocate(
		input.engine_handle(),
		vec![geometry.rows as usize, geometry.output_features as usize],
		output_count,
		DType::F32,
	)?;
	let buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::read(weight.storage()),
		BufferBinding::read(offsets.storage()),
		BufferBinding::write(output.storage()),
	];
	let push_constants = geometry.push_constants();
	let kernel = KernelId::MlGroupedGemmMF32;
	record_semantic(
		&[input, weight, offsets],
		&[&output],
		&[],
		kernel,
		&buffers,
		&push_constants,
		[
			geometry.rows.div_ceil(32),
			geometry.output_features.div_ceil(32),
			1,
		],
	)?;
	Ok(output)
}

pub(in crate::ml) fn grouped_gemm_m_backward(
	output_gradient: &Matrix,
	input: &Matrix,
	weight: &Matrix,
	offsets: &Matrix,
) -> Result<GroupedGemmBackward> {
	const OPERATION: &str = crate::core::operation::ml::GROUPED_GEMM_M_BACKWARD.name();
	let geometry = GroupedLinearGeometry::resolve(input, weight, None, offsets, OPERATION)?;
	if output_gradient.shape() != [geometry.rows as usize, geometry.output_features as usize]
		|| output_gradient.dtype() != DType::F32
		|| !input
			.engine_handle()
			.same_as(output_gradient.engine_handle())
	{
		return Err(Error::invalid_argument(format!(
			"{OPERATION} output gradient must be same-engine F32 [R, N]"
		)));
	}
	let input_gradient = Matrix::allocate(
		input.engine_handle(),
		input.shape().to_vec(),
		input.num_elements(),
		DType::F32,
	)?;
	let weight_gradient = Matrix::allocate(
		input.engine_handle(),
		weight.shape().to_vec(),
		weight.num_elements(),
		DType::F32,
	)?;
	let push_constants = geometry.push_constants();
	let data_buffers = [
		BufferBinding::read(output_gradient.storage()),
		BufferBinding::read(weight.storage()),
		BufferBinding::read(offsets.storage()),
		BufferBinding::write(input_gradient.storage()),
	];
	let parameter_buffers = [
		BufferBinding::read(output_gradient.storage()),
		BufferBinding::read(input.storage()),
		BufferBinding::read(offsets.storage()),
		BufferBinding::write(weight_gradient.storage()),
	];
	let dispatches = [
		ComputeDispatch {
			kernel: KernelId::MlGroupedGemmMDataBackwardF32,
			buffers: &data_buffers,
			push_constants: &push_constants,
			workgroups: [
				geometry.rows.div_ceil(32),
				geometry.input_features.div_ceil(32),
				1,
			],
		},
		ComputeDispatch {
			kernel: KernelId::MlGroupedGemmMParameterBackwardF32,
			buffers: &parameter_buffers,
			push_constants: &push_constants,
			workgroups: [
				geometry.output_features.div_ceil(32),
				geometry.input_features.div_ceil(32),
				geometry.experts,
			],
		},
	];
	let inputs = [output_gradient, input, weight, offsets];
	let outputs = [&input_gradient, &weight_gradient];
	input.engine_handle().record_split_semantic(
		&dispatches,
		SemanticDispatch {
			contract: crate::core::operation::ml::GROUPED_GEMM_M_BACKWARD,
			inputs: &inputs,
			outputs: &outputs,
			attributes: &[],
		},
	)?;
	Ok(GroupedGemmBackward {
		input: input_gradient,
		weight: weight_gradient,
	})
}

pub(in crate::ml) fn grouped_linear_m(
	input: &Matrix,
	weight: &Matrix,
	bias: &Matrix,
	offsets: &Matrix,
) -> Result<Matrix> {
	const OPERATION: &str = crate::core::operation::ml::GROUPED_LINEAR_M.name();
	let geometry = GroupedLinearGeometry::resolve(input, weight, Some(bias), offsets, OPERATION)?;
	let output_count = (geometry.rows as usize)
		.checked_mul(geometry.output_features as usize)
		.ok_or_else(|| {
			Error::invalid_argument(format!("{OPERATION} output size overflows usize"))
		})?;
	let output = Matrix::allocate(
		input.engine_handle(),
		vec![geometry.rows as usize, geometry.output_features as usize],
		output_count,
		DType::F32,
	)?;
	let buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::read(weight.storage()),
		BufferBinding::read(bias.storage()),
		BufferBinding::read(offsets.storage()),
		BufferBinding::write(output.storage()),
	];
	let push_constants = geometry.push_constants();
	let kernel = KernelId::MlGroupedLinearMF32;
	record_semantic(
		&[input, weight, bias, offsets],
		&[&output],
		&[],
		kernel,
		&buffers,
		&push_constants,
		[
			geometry.rows.div_ceil(32),
			geometry.output_features.div_ceil(32),
			1,
		],
	)?;
	Ok(output)
}

pub(in crate::ml) fn grouped_linear_m_backward(
	output_gradient: &Matrix,
	input: &Matrix,
	weight: &Matrix,
	offsets: &Matrix,
) -> Result<GroupedLinearBackward> {
	const OPERATION: &str = crate::core::operation::ml::GROUPED_LINEAR_M_BACKWARD.name();
	let geometry = GroupedLinearGeometry::resolve(input, weight, None, offsets, OPERATION)?;
	if output_gradient.shape() != [geometry.rows as usize, geometry.output_features as usize]
		|| output_gradient.dtype() != DType::F32
		|| !input
			.engine_handle()
			.same_as(output_gradient.engine_handle())
	{
		return Err(Error::invalid_argument(format!(
			"{OPERATION} output gradient must be same-engine F32 [R, N]"
		)));
	}

	let input_gradient = Matrix::allocate(
		input.engine_handle(),
		input.shape().to_vec(),
		input.num_elements(),
		DType::F32,
	)?;
	let weight_gradient = Matrix::allocate(
		input.engine_handle(),
		weight.shape().to_vec(),
		weight.num_elements(),
		DType::F32,
	)?;
	let bias_count = (geometry.experts as usize)
		.checked_mul(geometry.output_features as usize)
		.ok_or_else(|| Error::invalid_argument(format!("{OPERATION} bias size overflows usize")))?;
	let bias_gradient = Matrix::allocate(
		input.engine_handle(),
		vec![geometry.experts as usize, geometry.output_features as usize],
		bias_count,
		DType::F32,
	)?;
	let push_constants = geometry.push_constants();
	let data_buffers = [
		BufferBinding::read(output_gradient.storage()),
		BufferBinding::read(weight.storage()),
		BufferBinding::read(offsets.storage()),
		BufferBinding::write(input_gradient.storage()),
	];
	let parameter_buffers = [
		BufferBinding::read(output_gradient.storage()),
		BufferBinding::read(input.storage()),
		BufferBinding::read(offsets.storage()),
		BufferBinding::write(weight_gradient.storage()),
		BufferBinding::write(bias_gradient.storage()),
	];
	let dispatches = [
		ComputeDispatch {
			kernel: KernelId::MlGroupedLinearMDataBackwardF32,
			buffers: &data_buffers,
			push_constants: &push_constants,
			workgroups: [
				geometry.rows.div_ceil(32),
				geometry.input_features.div_ceil(32),
				1,
			],
		},
		ComputeDispatch {
			kernel: KernelId::MlGroupedLinearMParameterBackwardF32,
			buffers: &parameter_buffers,
			push_constants: &push_constants,
			workgroups: [
				geometry.output_features.div_ceil(32),
				geometry.input_features.div_ceil(32),
				geometry.experts,
			],
		},
	];
	let inputs = [output_gradient, input, weight, offsets];
	let outputs = [&input_gradient, &weight_gradient, &bias_gradient];
	input.engine_handle().record_split_semantic(
		&dispatches,
		SemanticDispatch {
			contract: crate::core::operation::ml::GROUPED_LINEAR_M_BACKWARD,
			inputs: &inputs,
			outputs: &outputs,
			attributes: &[],
		},
	)?;
	Ok(GroupedLinearBackward {
		input: input_gradient,
		weight: weight_gradient,
		bias: bias_gradient,
	})
}
