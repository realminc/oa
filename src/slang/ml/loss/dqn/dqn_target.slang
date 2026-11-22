// Mechanical adaptation of OA loss/rlDqnTarget.slang.

import attributes;
import storage;

struct PushConstants {
	uint reward_index;
	uint next_q_index;
	uint terminated_index;
	uint truncated_index;
	uint output_index;
	uint batch;
	uint actions;
	float discount;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("dqn_target")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID) {
	uint row = dispatch_thread_id.x;
	if (row >= push.batch) {
		return;
	}

	float next_q = -3.402823466e+38f;
	for (uint action = 0u; action < push.actions; ++action) {
		next_q = max(
			next_q,
			load_f32(storage_buffers[push.next_q_index], row * push.actions + action));
	}
	float reward = load_f32(storage_buffers[push.reward_index], row);
	bool terminated = load_u8(storage_buffers[push.terminated_index], row) != 0u;
	// push.truncated_index deliberately does not suppress bootstrap: truncation
	// stops an episode trace while retaining the next-state value.
	float bootstrap = terminated ? 0.0f : push.discount * next_q;
	store_f32(storage_buffers[push.output_index], row, reward + bootstrap);
}
