// Functional momentum update for BatchNorm2d persistent running statistics.

import storage;
import attributes;

struct PushConstants {
	uint running_mean_index;
	uint running_variance_index;
	uint batch_mean_index;
	uint batch_variance_index;
	uint updated_mean_index;
	uint updated_variance_index;
	uint channels;
	float momentum;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("batch_norm_2d_running_update")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID) {
	uint channel = dispatch_thread_id.x;
	if (channel >= push.channels) {
		return;
	}
	float keep = 1.0f - push.momentum;
	float running_mean = load_f32(storage_buffers[push.running_mean_index], channel);
	float running_variance = load_f32(storage_buffers[push.running_variance_index], channel);
	float batch_mean = load_f32(storage_buffers[push.batch_mean_index], channel);
	float batch_variance = load_f32(storage_buffers[push.batch_variance_index], channel);
	store_f32(
		storage_buffers[push.updated_mean_index],
		channel,
		running_mean * keep + batch_mean * push.momentum
	);
	store_f32(
		storage_buffers[push.updated_variance_index],
		channel,
		running_variance * keep + batch_variance * push.momentum
	);
}
