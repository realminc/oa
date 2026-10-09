import storage;
import attributes;

struct PushConstants {
	uint summary_index;
	uint state_index;
	uint channels;
	uint blocks;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("biquad_scan")]
[domain("audio")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint channel = thread_id.x;
	if (channel >= push.channels) return;
	float state0 = 0.0f, state1 = 0.0f;
	for (uint block = 0; block < push.blocks; ++block) {
		uint task = channel * push.blocks + block;
		store_f32(storage_buffers[push.state_index], task * 2 + 0, state0);
		store_f32(storage_buffers[push.state_index], task * 2 + 1, state1);
		uint summary = task * 6;
		float m00 = load_f32(storage_buffers[push.summary_index], summary + 0);
		float m01 = load_f32(storage_buffers[push.summary_index], summary + 1);
		float m10 = load_f32(storage_buffers[push.summary_index], summary + 2);
		float m11 = load_f32(storage_buffers[push.summary_index], summary + 3);
		float v0 = load_f32(storage_buffers[push.summary_index], summary + 4);
		float v1 = load_f32(storage_buffers[push.summary_index], summary + 5);
		float next0 = m00 * state0 + m01 * state1 + v0;
		float next1 = m10 * state0 + m11 * state1 + v1;
		state0 = next0; state1 = next1;
	}
}
