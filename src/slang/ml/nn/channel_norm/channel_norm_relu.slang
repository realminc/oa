// Ported from OA C++ channelNorm/channelNormRelu.slang.
import storage;
import attributes;

struct PushConstants {
	uint input_index;
	uint weight_index;
	uint bias_index;
	uint output_index;
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

[kernel_name("channel_norm_relu")]
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
	float values[4];
	uint count = 0;
	for (uint channel = lane; channel < push.channels; channel += 256) {
		values[count++] = load_f32(storage_buffers[push.input_index], base + channel * push.sequence_length);
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
	uint element = base + channel * push.sequence_length;
	for (uint index = 0; index < count; ++index, channel += 256, element += 256 * push.sequence_length) {
		float normalized = (values[index] - mean) * inverse_stddev;
		float output = normalized * load_f32(storage_buffers[push.weight_index], channel)
			+ load_f32(storage_buffers[push.bias_index], channel);
		store_f32(storage_buffers[push.output_index], element, max(0.0f, output));
	}
}
