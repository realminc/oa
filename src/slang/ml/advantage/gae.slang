// Mechanical adaptation of OA advantage/gae.slang.

import storage;
import attributes;

struct PushConstants {
	uint reward_index;
	uint value_index;
	uint next_value_index;
	uint terminated_index;
	uint truncated_index;
	uint advantage_index;
	uint return_index;
	uint time;
	uint environments;
	float gamma;
	float lambda;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("gae")]
[domain("ml")]
[variant("reverse_time")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID) {
	uint environment = dispatch_thread_id.x;
	if (environment >= push.environments) {
		return;
	}

	float next_advantage = 0.0f;
	for (uint reverse_time = push.time; reverse_time > 0u; --reverse_time) {
		uint index = (reverse_time - 1u) * push.environments + environment;
		bool terminated = load_u8(storage_buffers[push.terminated_index], index) != 0u;
		bool truncated = load_u8(storage_buffers[push.truncated_index], index) != 0u;
		float bootstrap_mask = terminated ? 0.0f : 1.0f;
		float trace_mask = (terminated || truncated) ? 0.0f : 1.0f;
		float value = load_f32(storage_buffers[push.value_index], index);
		float delta = load_f32(storage_buffers[push.reward_index], index)
			+ push.gamma * bootstrap_mask
				* load_f32(storage_buffers[push.next_value_index], index)
			- value;
		float advantage = delta
			+ push.gamma * push.lambda * trace_mask * next_advantage;
		store_f32(storage_buffers[push.advantage_index], index, advantage);
		store_f32(storage_buffers[push.return_index], index, advantage + value);
		next_advantage = advantage;
	}
}
