// FP32 BatchNorm2d training statistics. One workgroup owns one NCHW channel.

import storage;
import attributes;

struct PushConstants {
	uint input_index;
	uint mean_index;
	uint variance_index;
	uint batch_size;
	uint channels;
	uint height;
	uint width;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

groupshared float partial[256];
groupshared float channel_mean;

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

[kernel_name("batch_norm_2d")]
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
	float local_sum = 0.0f;
	for (uint sample = lane; sample < sample_count; sample += 256u) {
		uint batch = sample / spatial;
		uint spatial_index = sample % spatial;
		uint index = (batch * push.channels + channel) * spatial + spatial_index;
		local_sum += load_f32(storage_buffers[push.input_index], index);
	}
	float sum = reduce_sum(local_sum, lane);
	if (lane == 0u) {
		channel_mean = sum / float(sample_count);
		store_f32(storage_buffers[push.mean_index], channel, channel_mean);
	}
	GroupMemoryBarrierWithGroupSync();

	float local_square_sum = 0.0f;
	for (uint sample = lane; sample < sample_count; sample += 256u) {
		uint batch = sample / spatial;
		uint spatial_index = sample % spatial;
		uint index = (batch * push.channels + channel) * spatial + spatial_index;
		float centered = load_f32(storage_buffers[push.input_index], index) - channel_mean;
		local_square_sum += centered * centered;
	}
	float square_sum = reduce_sum(local_square_sum, lane);
	if (lane == 0u) {
		store_f32(
			storage_buffers[push.variance_index],
			channel,
			max(square_sum / float(sample_count), 0.0f)
		);
	}
}
