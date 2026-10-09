// Mechanical adaptation of OA loss/rlPpoClipBwd.slang.

import attributes;
import storage;

struct PushConstants {
	uint new_log_probability_index;
	uint old_log_probability_index;
	uint advantage_index;
	uint gradient_index;
	uint count;
	float clip_epsilon;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("ppo_clipped_policy_backward")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID) {
	uint index = dispatch_thread_id.x;
	if (index >= push.count) {
		return;
	}
	float ratio = exp(
		load_f32(storage_buffers[push.new_log_probability_index], index)
		- load_f32(storage_buffers[push.old_log_probability_index], index));
	float advantage = load_f32(storage_buffers[push.advantage_index], index);
	bool clipping_active = (advantage >= 0.0 && ratio > 1.0 + push.clip_epsilon)
		|| (advantage < 0.0 && ratio < 1.0 - push.clip_epsilon);
	float gradient = clipping_active ? 0.0 : -(advantage * ratio) / float(push.count);
	store_f32(storage_buffers[push.gradient_index], index, gradient);
}
