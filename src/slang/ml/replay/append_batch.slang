// Mechanical adaptation of OA replay/rlReplayAppend.slang.

import storage;
import attributes;

struct PushConstants {
	uint observation_index;
	uint action_index;
	uint next_observation_index;
	uint reward_index;
	uint terminated_index;
	uint truncated_index;
	uint output_observation_index;
	uint output_action_index;
	uint output_next_observation_index;
	uint output_reward_index;
	uint output_terminated_index;
	uint output_truncated_index;
	uint cursor;
	uint capacity;
	uint batch;
	uint observation_elements;
	uint action_elements;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("append_batch")]
[domain("ml")]
[variant("circular")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID) {
	uint index = dispatch_thread_id.x;
	uint observation_count = push.batch * push.observation_elements;
	if (index < observation_count) {
		uint lane = index / push.observation_elements;
		uint component = index - lane * push.observation_elements;
		uint destination = ((push.cursor + lane) % push.capacity)
			* push.observation_elements + component;
		store_f32(
			storage_buffers[push.output_observation_index], destination,
			load_f32(storage_buffers[push.observation_index], index));
		store_f32(
			storage_buffers[push.output_next_observation_index], destination,
			load_f32(storage_buffers[push.next_observation_index], index));
	}
	uint action_count = push.batch * push.action_elements;
	if (index < action_count) {
		uint lane = index / push.action_elements;
		uint component = index - lane * push.action_elements;
		uint destination = ((push.cursor + lane) % push.capacity)
			* push.action_elements + component;
		storage_buffers[push.output_action_index].Store(
			destination * 4u,
			storage_buffers[push.action_index].Load(index * 4u));
	}
	if (index >= push.batch) {
		return;
	}
	uint destination = (push.cursor + index) % push.capacity;
	store_f32(
		storage_buffers[push.output_reward_index], destination,
		load_f32(storage_buffers[push.reward_index], index));
	store_u8(
		storage_buffers[push.output_terminated_index], destination,
		load_u8(storage_buffers[push.terminated_index], index));
	store_u8(
		storage_buffers[push.output_truncated_index], destination,
		load_u8(storage_buffers[push.truncated_index], index));
}
