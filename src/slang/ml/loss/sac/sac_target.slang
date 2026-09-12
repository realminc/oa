// Mechanical adaptation of OA loss/rlSacTarget.slang.

import attributes;
import storage;

struct PushConstants {
	uint reward_index;
	uint next_q1_index;
	uint next_q2_index;
	uint next_log_probability_index;
	uint terminated_index;
	uint truncated_index;
	uint output_index;
	uint batch;
	float discount;
	float entropy_coefficient;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("sac_target")]
[domain("ml")]
[variant("twin_q")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID) {
	uint index = dispatch_thread_id.x;
	if (index >= push.batch) {
		return;
	}

	float reward = load_f32(storage_buffers[push.reward_index], index);
	float next_q1 = load_f32(storage_buffers[push.next_q1_index], index);
	float next_q2 = load_f32(storage_buffers[push.next_q2_index], index);
	float next_log_probability = load_f32(
		storage_buffers[push.next_log_probability_index], index);
	bool terminated = load_u8(storage_buffers[push.terminated_index], index) != 0u;
	// push.truncated_index deliberately does not suppress bootstrap.
	float bootstrap = min(next_q1, next_q2)
		- push.entropy_coefficient * next_log_probability;
	float target = reward + (terminated ? 0.0f : push.discount * bootstrap);
	store_f32(storage_buffers[push.output_index], index, target);
}
