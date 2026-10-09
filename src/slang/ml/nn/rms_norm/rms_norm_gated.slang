// Fused broadcast-affine RMSNorm followed by a SiLU gate.
// Donor: oa/ml/shader/compute/nn/rmsnorm/rmsNormGated.slang

import storage;
import activations;
import attributes;

struct PushConstants {
	uint input_index;
	uint weight_index;
	uint bias_index;
	uint gate_index;
	uint output_index;
	uint rows;
	uint columns;
	uint groups;
	float epsilon;
	uint has_bias;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

groupshared float partial[256];

[kernel_name("rms_norm_gated")]
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
	uint affine_base = (row % push.groups) * push.columns;
	float local_square_sum = 0.0f;
	for (uint column = lane; column < push.columns; column += 256u) {
		float value = load_f32(storage_buffers[push.input_index], base + column);
		local_square_sum += value * value;
	}
	partial[lane] = local_square_sum;
	GroupMemoryBarrierWithGroupSync();
	for (uint stride = 128u; stride > 0u; stride >>= 1u) {
		if (lane < stride) partial[lane] += partial[lane + stride];
		GroupMemoryBarrierWithGroupSync();
	}
	float inverse_rms = 1.0f / sqrt(
		partial[0] / float(push.columns) + push.epsilon);

	for (uint column = lane; column < push.columns; column += 256u) {
		uint index = base + column;
		float normalized = load_f32(storage_buffers[push.input_index], index)
			* inverse_rms
			* load_f32(storage_buffers[push.weight_index], affine_base + column);
		if (push.has_bias != 0u) {
			normalized += load_f32(
				storage_buffers[push.bias_index], affine_base + column);
		}
		float gate = load_f32(storage_buffers[push.gate_index], index);
		store_f32(storage_buffers[push.output_index], index, normalized * silu(gate));
	}
}
