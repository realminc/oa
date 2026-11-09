// Submit-time fusion of the Conv1d bias add, NCL transpose and the
// following channel normalization.
// Donor: src/slang/ml/nn/channel_norm/channel_norm.slang
// The conv output is row-major in the projected [B*T, C] GEMM buffer:
// the normalization domain (all C at one batch/time position) is one
// contiguous row, so statistics keep the donor's lane-strided order on
// the biased value. Both the biased conv output (the backward pass's
// saved input) and the normalized output are republished in [B, C, T].

import storage;
import attributes;

struct PushConstants {
	uint input_index;
	uint conv_bias_index;
	uint weight_index;
	uint bias_index;
	uint output_index;
	uint conv_output_index;
	uint batch;
	uint channels;
	uint sequence_length;
	float epsilon;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];
groupshared float reduction[256];

float reduce_sum(float value, uint lane) {
	reduction[lane] = value;
	GroupMemoryBarrierWithGroupSync();
	for (uint stride = 128; stride > 0; stride >>= 1) {
		if (lane < stride) {
			reduction[lane] += reduction[lane + stride];
		}
		GroupMemoryBarrierWithGroupSync();
	}
	return reduction[0];
}

[kernel_name("conv_1d_bias_transpose_norm")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint3 thread_id : SV_GroupThreadID) {
	uint row = group_id.x;
	if (row >= push.batch * push.sequence_length) {
		return;
	}
	uint lane = thread_id.x;
	uint row_base = row * push.channels;
	uint conv_base = (row / push.sequence_length) * push.channels * push.sequence_length
		+ row % push.sequence_length;
	float values[4];
	uint count = 0;
	for (uint channel = lane; channel < push.channels; channel += 256) {
		values[count++] = load_f32(storage_buffers[push.input_index], row_base + channel)
			+ load_f32(storage_buffers[push.conv_bias_index], channel);
	}
	float local_sum = 0.0f;
	for (uint index = 0; index < count; ++index) {
		local_sum += values[index];
	}
	float mean = reduce_sum(local_sum, lane) / float(push.channels);
	GroupMemoryBarrierWithGroupSync();
	float local_variance = 0.0f;
	for (uint index = 0; index < count; ++index) {
		float difference = values[index] - mean;
		local_variance += difference * difference;
	}
	float inverse_stddev = rsqrt(reduce_sum(local_variance, lane) / float(push.channels) + push.epsilon);
	uint channel = lane;
	uint element = conv_base + channel * push.sequence_length;
	for (uint index = 0; index < count; ++index, channel += 256, element += 256 * push.sequence_length) {
		float normalized = (values[index] - mean) * inverse_stddev;
		float output = normalized * load_f32(storage_buffers[push.weight_index], channel)
			+ load_f32(storage_buffers[push.bias_index], channel);
		store_f32(storage_buffers[push.output_index], element, output);
		store_f32(storage_buffers[push.conv_output_index], element, values[index]);
	}
}
