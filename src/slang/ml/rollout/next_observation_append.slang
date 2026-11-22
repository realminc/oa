// Compiled-rollout next_observation retention. The cursor-addressed row copy
// preserves each transition's pre-reset successor observation for one batched
// value evaluation in the GAE plan; done lanes lose that state when the
// environment reset overwrites the live observation buffer in place.

import storage;
import attributes;

struct PushConstants {
	uint output_index;
	uint next_observation_index;
	uint cursor_index;
	uint step;
	uint observation_count;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("next_observation_append")]
[domain("ml")]
[variant("cursor_row")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID) {
	uint index = dispatch_thread_id.x;
	if (index >= push.observation_count) {
		return;
	}
	uint output_index =
		(push.step + load_u32(storage_buffers[push.cursor_index], 0)) *
			push.observation_count +
		index;
	store_f32(
		storage_buffers[push.output_index], output_index,
		load_f32(storage_buffers[push.next_observation_index], index));
}
