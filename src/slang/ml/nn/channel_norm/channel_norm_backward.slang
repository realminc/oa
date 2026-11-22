// Ported from OA C++ channelNorm/channelNormBwd.slang.
import storage;
import attributes;

struct PushConstants {
	uint input_index;
	uint weight_index;
	uint output_gradient_index;
	uint input_gradient_index;
	uint weight_contribution_index;
	uint bias_contribution_index;
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

[kernel_name("channel_norm_backward")]
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
	uint batch_index = row / push.sequence_length;
	uint sequence_index = row % push.sequence_length;
	uint lane = thread_id.x;
	uint base = batch_index * push.channels * push.sequence_length + sequence_index;
	float input_values[4];
	float gradient_values[4];
	float weight_values[4];
	uint count = 0;
	for (uint channel = lane; channel < push.channels; channel += 256) {
		uint element = base + channel * push.sequence_length;
		input_values[count] = load_f32(storage_buffers[push.input_index], element);
		gradient_values[count] = load_f32(storage_buffers[push.output_gradient_index], element);
		weight_values[count] = load_f32(storage_buffers[push.weight_index], channel);
		++count;
	}
	float local_sum = 0.0f;
	for (uint index = 0; index < count; ++index) local_sum += input_values[index];
	float mean = reduce_sum(local_sum, lane) / float(push.channels);
	GroupMemoryBarrierWithGroupSync();
	float local_variance = 0.0f;
	for (uint index = 0; index < count; ++index) {
		float difference = input_values[index] - mean;
		local_variance += difference * difference;
	}
	float inverse_stddev = rsqrt(reduce_sum(local_variance, lane) / float(push.channels) + push.epsilon);
	GroupMemoryBarrierWithGroupSync();
	float local_dxhat = 0.0f;
	float local_normalized_dxhat = 0.0f;
	for (uint index = 0; index < count; ++index) {
		float normalized = (input_values[index] - mean) * inverse_stddev;
		float dxhat = gradient_values[index] * weight_values[index];
		local_dxhat += dxhat;
		local_normalized_dxhat += normalized * dxhat;
	}
	float sum_dxhat = reduce_sum(local_dxhat, lane);
	GroupMemoryBarrierWithGroupSync();
	float sum_normalized_dxhat = reduce_sum(local_normalized_dxhat, lane);
	GroupMemoryBarrierWithGroupSync();
	float mean_dxhat = sum_dxhat / float(push.channels);
	float mean_normalized_dxhat = sum_normalized_dxhat / float(push.channels);
	uint contribution_base = row * push.channels;
	uint channel = lane;
	uint element = base + channel * push.sequence_length;
	for (uint index = 0; index < count; ++index, channel += 256, element += 256 * push.sequence_length) {
		float normalized = (input_values[index] - mean) * inverse_stddev;
		float gradient = gradient_values[index];
		float dxhat = gradient * weight_values[index];
		store_f32(storage_buffers[push.input_gradient_index], element,
			inverse_stddev * (dxhat - mean_dxhat - normalized * mean_normalized_dxhat));
		store_f32(storage_buffers[push.weight_contribution_index], contribution_base + channel, gradient * normalized);
		store_f32(storage_buffers[push.bias_contribution_index], contribution_base + channel, gradient);
	}
}
