import storage;
import attributes;

struct PushConstants {
	uint input_index;
	uint state_index;
	uint output_index;
	uint channels;
	uint samples;
	uint blocks;
	uint block_size;
	float b0;
	float b1;
	float b2;
	float a1;
	float a2;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("biquad_apply")]
[domain("audio")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint task = thread_id.x;
	if (task >= push.channels * push.blocks) return;
	uint channel = task / push.blocks;
	uint block = task - channel * push.blocks;
	uint first = block * push.block_size;
	uint last = min(first + push.block_size, push.samples);
	uint base = channel * push.samples;
	float state0 = load_f32(storage_buffers[push.state_index], task * 2 + 0);
	float state1 = load_f32(storage_buffers[push.state_index], task * 2 + 1);
	for (uint sample = first; sample < last; ++sample) {
		uint index = base + sample;
		float x = load_f32(storage_buffers[push.input_index], index);
		float y = push.b0 * x + state0;
		float next0 = push.b1 * x - push.a1 * y + state1;
		float next1 = push.b2 * x - push.a2 * y;
		store_f32(storage_buffers[push.output_index], index, y);
		state0 = next0; state1 = next1;
	}
}
