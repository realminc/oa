// Mechanical adaptation of OA rollout/rlRolloutAppend.slang.

import storage;
import attributes;

struct PushConstants {
	uint observation_index;
	uint action_index;
	uint reward_index;
	uint value_index;
	uint next_value_index;
	uint log_probability_index;
	uint terminated_index;
	uint truncated_index;
	uint output_observation_index;
	uint output_action_index;
	uint output_reward_index;
	uint output_value_index;
	uint output_next_value_index;
	uint output_log_probability_index;
	uint output_terminated_index;
	uint output_truncated_index;
	uint output_valid_index;
	uint step;
	uint environments;
	uint observation_elements;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("append")]
[domain("ml")]
[variant("categorical_time_major")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID) {
	uint index = dispatch_thread_id.x;
	uint observation_count = push.environments * push.observation_elements;
	if (index < observation_count) {
		uint output_index = push.step * observation_count + index;
		store_f32(
			storage_buffers[push.output_observation_index], output_index,
			load_f32(storage_buffers[push.observation_index], index));
	}
	if (index >= push.environments) {
		return;
	}

	uint output_index = push.step * push.environments + index;
	store_i32(
		storage_buffers[push.output_action_index], output_index,
		load_i32(storage_buffers[push.action_index], index));
	store_f32(
		storage_buffers[push.output_reward_index], output_index,
		load_f32(storage_buffers[push.reward_index], index));
	store_f32(
		storage_buffers[push.output_value_index], output_index,
		load_f32(storage_buffers[push.value_index], index));
	store_f32(
		storage_buffers[push.output_next_value_index], output_index,
		load_f32(storage_buffers[push.next_value_index], index));
	store_f32(
		storage_buffers[push.output_log_probability_index], output_index,
		load_f32(storage_buffers[push.log_probability_index], index));
	store_u8(
		storage_buffers[push.output_terminated_index], output_index,
		load_u8(storage_buffers[push.terminated_index], index));
	store_u8(
		storage_buffers[push.output_truncated_index], output_index,
		load_u8(storage_buffers[push.truncated_index], index));
	store_u8(storage_buffers[push.output_valid_index], output_index, 1u);
}
