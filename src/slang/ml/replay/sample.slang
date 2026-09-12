// Mechanical adaptation of OA replay/rlReplaySample.slang.

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
	uint output_index_index;
	uint size;
	uint batch;
	uint observation_elements;
	uint action_elements;
	uint seed_low;
	uint seed_high;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[ForceInline]
uint hash32(uint value) {
	value ^= value >> 16u;
	value *= 0x7feb352du;
	value ^= value >> 15u;
	value *= 0x846ca68bu;
	return value ^ (value >> 16u);
}

[ForceInline]
uint source_index(uint lane) {
	return hash32(push.seed_low ^ hash32(push.seed_high + lane * 0x9e3779b9u))
		% push.size;
}

[kernel_name("sample")]
[domain("ml")]
[variant("deterministic_uniform")]
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
		uint source = source_index(lane) * push.observation_elements + component;
		store_f32(
			storage_buffers[push.output_observation_index], index,
			load_f32(storage_buffers[push.observation_index], source));
		store_f32(
			storage_buffers[push.output_next_observation_index], index,
			load_f32(storage_buffers[push.next_observation_index], source));
	}
	uint action_count = push.batch * push.action_elements;
	if (index < action_count) {
		uint lane = index / push.action_elements;
		uint component = index - lane * push.action_elements;
		uint source = source_index(lane) * push.action_elements + component;
		storage_buffers[push.output_action_index].Store(
			index * 4u, storage_buffers[push.action_index].Load(source * 4u));
	}
	if (index >= push.batch) {
		return;
	}
	uint source = source_index(index);
	store_f32(
		storage_buffers[push.output_reward_index], index,
		load_f32(storage_buffers[push.reward_index], source));
	store_u8(
		storage_buffers[push.output_terminated_index], index,
		load_u8(storage_buffers[push.terminated_index], source));
	store_u8(
		storage_buffers[push.output_truncated_index], index,
		load_u8(storage_buffers[push.truncated_index], source));
	store_u32(storage_buffers[push.output_index_index], index, source);
}
