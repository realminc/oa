// FP32 BatchNorm2d normalization using explicit per-channel statistics.

import storage;
import attributes;

struct PushConstants {
	uint input_index;
	uint mean_index;
	uint variance_index;
	uint weight_index;
	uint bias_index;
	uint output_index;
	uint batch_size;
	uint channels;
	uint height;
	uint width;
	float epsilon;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("batch_norm_2d_with_stats")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID) {
	uint x = group_id.x * 16u + group_thread_id.x;
	uint y = group_id.y * 16u + group_thread_id.y;
	uint batch_channel = group_id.z;
	if (x >= push.width || y >= push.height) {
		return;
	}
	uint batch = batch_channel / push.channels;
	uint channel = batch_channel % push.channels;
	if (batch >= push.batch_size) {
		return;
	}

	uint spatial = push.height * push.width;
	uint index = (batch * push.channels + channel) * spatial + y * push.width + x;
	float mean = load_f32(storage_buffers[push.mean_index], channel);
	float variance = load_f32(storage_buffers[push.variance_index], channel);
	float weight = load_f32(storage_buffers[push.weight_index], channel);
	float bias = load_f32(storage_buffers[push.bias_index], channel);
	float value = load_f32(storage_buffers[push.input_index], index);
	float normalized = (value - mean) * rsqrt(max(variance, 0.0f) + push.epsilon);
	store_f32(storage_buffers[push.output_index], index, normalized * weight + bias);
}
