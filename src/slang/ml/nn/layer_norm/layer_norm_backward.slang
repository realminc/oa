// Row-cooperative LayerNorm data adjoint from saved normalized values.

import storage;
import attributes;

struct PushConstants {
	uint normalized_index;
	uint inverse_stddev_index;
	uint weight_index;
	uint output_gradient_index;
	uint input_gradient_index;
	uint weight_contribution_index;
	uint rows;
	uint columns;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

groupshared float partial[256];
groupshared float sum_scaled_gradient;
groupshared float sum_scaled_gradient_normalized;

float reduce_sum(float value, uint lane) {
	partial[lane] = value;
	GroupMemoryBarrierWithGroupSync();
	for (uint stride = 128u; stride > 0u; stride >>= 1u) {
		if (lane < stride) {
			partial[lane] += partial[lane + stride];
		}
		GroupMemoryBarrierWithGroupSync();
	}
	return partial[0];
}

[kernel_name("layer_norm_backward")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID) {
	uint row = group_id.x;
	uint lane = group_thread_id.x;
	if (row >= push.rows) {
		return;
	}
	uint base = row * push.columns;
	float local_scaled_gradient = 0.0f;
	float local_scaled_gradient_normalized = 0.0f;
	for (uint column = lane; column < push.columns; column += 256u) {
		float scaled_gradient = load_f32(
			storage_buffers[push.output_gradient_index], base + column
		) * load_f32(storage_buffers[push.weight_index], column);
		local_scaled_gradient += scaled_gradient;
		local_scaled_gradient_normalized += scaled_gradient * load_f32(
			storage_buffers[push.normalized_index], base + column
		);
	}
	float scaled_gradient_total = reduce_sum(local_scaled_gradient, lane);
	if (lane == 0u) {
		sum_scaled_gradient = scaled_gradient_total;
	}
	GroupMemoryBarrierWithGroupSync();
	float scaled_gradient_normalized_total = reduce_sum(
		local_scaled_gradient_normalized,
		lane
	);
	if (lane == 0u) {
		sum_scaled_gradient_normalized = scaled_gradient_normalized_total;
	}
	GroupMemoryBarrierWithGroupSync();

	float inverse_stddev = load_f32(storage_buffers[push.inverse_stddev_index], row);
	float inverse_columns = 1.0f / float(push.columns);
	for (uint column = lane; column < push.columns; column += 256u) {
		float output_gradient = load_f32(
			storage_buffers[push.output_gradient_index], base + column
		);
		float weight = load_f32(storage_buffers[push.weight_index], column);
		float normalized = load_f32(storage_buffers[push.normalized_index], base + column);
		float input_gradient = inverse_stddev * (
			output_gradient * weight
			- sum_scaled_gradient * inverse_columns
			- normalized * sum_scaled_gradient_normalized * inverse_columns
		);
		store_f32(storage_buffers[push.input_gradient_index], base + column, input_gradient);
		store_f32(
			storage_buffers[push.weight_contribution_index],
			base + column,
			output_gradient * normalized
		);
	}
}
