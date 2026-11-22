// Adapted from OA C++ rlRolloutAppend.slang and matrixCopyRegion.slang.
// Each invocation owns one observation element and, for index<count, one
// aligned scalar in all four statistic/action outputs. Actions remain I32.
import storage;
import attributes;
struct PushConstants {
	uint observation_index;
	uint action_index;
	uint old_log_probability_index;
	uint advantage_index;
	uint returns_index;
	uint output_observation_index;
	uint output_action_index;
	uint output_old_log_probability_index;
	uint output_advantage_index;
	uint output_returns_index;
	uint start;
	uint count;
	uint observation_elements;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];
[kernel_name("minibatch")]
[domain("ml")]
[variant("categorical_contiguous")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint i = tid.x;
	if (i < push.count * push.observation_elements) {
		store_f32(storage_buffers[push.output_observation_index], i,
			load_f32(storage_buffers[push.observation_index], push.start * push.observation_elements + i));
	}
	if (i >= push.count) return;
	uint source = push.start + i;
	store_i32(storage_buffers[push.output_action_index], i, load_i32(storage_buffers[push.action_index], source));
	store_f32(storage_buffers[push.output_old_log_probability_index], i, load_f32(storage_buffers[push.old_log_probability_index], source));
	store_f32(storage_buffers[push.output_advantage_index], i, load_f32(storage_buffers[push.advantage_index], source));
	store_f32(storage_buffers[push.output_returns_index], i, load_f32(storage_buffers[push.returns_index], source));
}
