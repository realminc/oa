// FP32 RMSNorm data adjoint adapted from OA's donor kernel.
// Donor: source/cpp/lib/oa/ml/shader/compute/nn/rmsnorm/rmsNormBwd.slang
// Shared reduction: source/cpp/lib/oa/core/shader/compute/math/reductions.slang

import storage;
import attributes;

struct PushConstants {
	uint input_index;
	uint weight_index;
	uint output_gradient_index;
	uint input_gradient_index;
	uint weight_contribution_index;
	uint rows;
	uint columns;
	float epsilon;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

groupshared float partial[256];

float reduce_sum(float value, uint lane) {
	partial[lane] = value;
	GroupMemoryBarrierWithGroupSync();
	for (uint stride = 128u; stride > 0u; stride >>= 1u) {
		if (lane < stride) partial[lane] += partial[lane + stride];
		GroupMemoryBarrierWithGroupSync();
	}
	return partial[0];
}

[kernel_name("rms_norm_backward")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID) {
	uint row = group_id.x;
	uint lane = group_thread_id.x;
	if (row >= push.rows) return;
	uint base = row * push.columns;

	float local_square_sum = 0.0f;
	for (uint column = lane; column < push.columns; column += 256u) {
		float value = load_f32(storage_buffers[push.input_index], base + column);
		local_square_sum += value * value;
	}
	float square_sum = reduce_sum(local_square_sum, lane);
	float inverse_rms = 1.0f / sqrt(square_sum / float(push.columns) + push.epsilon);
	GroupMemoryBarrierWithGroupSync();

	float local_dot = 0.0f;
	for (uint column = lane; column < push.columns; column += 256u) {
		uint index = base + column;
		local_dot += load_f32(storage_buffers[push.output_gradient_index], index)
			* load_f32(storage_buffers[push.weight_index], column)
			* load_f32(storage_buffers[push.input_index], index);
	}
	float dot = reduce_sum(local_dot, lane);
	float inverse_columns = 1.0f / float(push.columns);

	for (uint column = lane; column < push.columns; column += 256u) {
		uint index = base + column;
		float input = load_f32(storage_buffers[push.input_index], index);
		float output_gradient = load_f32(
			storage_buffers[push.output_gradient_index], index);
		float weight = load_f32(storage_buffers[push.weight_index], column);
		float input_gradient = inverse_rms * (
			output_gradient * weight
			- input * dot * inverse_rms * inverse_rms * inverse_columns);
		store_f32(storage_buffers[push.input_gradient_index], index, input_gradient);
		store_f32(storage_buffers[push.weight_contribution_index], index,
			output_gradient * input * inverse_rms);
	}
}
