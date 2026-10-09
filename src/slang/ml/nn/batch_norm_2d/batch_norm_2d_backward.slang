// BatchNorm2d parameter adjoints. One workgroup owns one NCHW channel.

import storage;
import attributes;

struct PushConstants {
	uint input_index;
	uint mean_index;
	uint variance_index;
	uint output_gradient_index;
	uint weight_gradient_index;
	uint bias_gradient_index;
	uint batch_size;
	uint channels;
	uint height;
	uint width;
	float epsilon;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

groupshared float partial_weight[256];
groupshared float partial_bias[256];

[kernel_name("batch_norm_2d_backward")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID) {
	uint channel = group_id.x;
	uint lane = group_thread_id.x;
	if (channel >= push.channels) {
		return;
	}
	uint spatial = push.height * push.width;
	uint sample_count = push.batch_size * spatial;
	float mean = load_f32(storage_buffers[push.mean_index], channel);
	float variance = load_f32(storage_buffers[push.variance_index], channel);
	float inverse_stddev = rsqrt(max(variance, 0.0f) + push.epsilon);
	float local_weight = 0.0f;
	float local_bias = 0.0f;
	for (uint sample = lane; sample < sample_count; sample += 256u) {
		uint batch = sample / spatial;
		uint spatial_index = sample % spatial;
		uint index = (batch * push.channels + channel) * spatial + spatial_index;
		float output_gradient = load_f32(storage_buffers[push.output_gradient_index], index);
		float input = load_f32(storage_buffers[push.input_index], index);
		local_weight += output_gradient * (input - mean) * inverse_stddev;
		local_bias += output_gradient;
	}
	partial_weight[lane] = local_weight;
	partial_bias[lane] = local_bias;
	GroupMemoryBarrierWithGroupSync();
	for (uint stride = 128u; stride > 0u; stride >>= 1u) {
		if (lane < stride) {
			partial_weight[lane] += partial_weight[lane + stride];
			partial_bias[lane] += partial_bias[lane + stride];
		}
		GroupMemoryBarrierWithGroupSync();
	}
	if (lane == 0u) {
		store_f32(storage_buffers[push.weight_gradient_index], channel, partial_weight[0]);
		store_f32(storage_buffers[push.bias_gradient_index], channel, partial_bias[0]);
	}
}
