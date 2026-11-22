import storage;
import attributes;

struct PushConstants {
	uint input_index;
	uint summary_index;
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

[kernel_name("biquad_summary")]
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
	uint input_base = channel * push.samples;
	float m00 = 1.0f, m01 = 0.0f, m10 = 0.0f, m11 = 1.0f;
	float v0 = 0.0f, v1 = 0.0f;
	for (uint sample = first; sample < last; ++sample) {
		float x = load_f32(storage_buffers[push.input_index], input_base + sample);
		float y = push.b0 * x + v0;
		float next_v0 = push.b1 * x - push.a1 * y + v1;
		float next_v1 = push.b2 * x - push.a2 * y;
		float next_m00 = -push.a1 * m00 + m10;
		float next_m01 = -push.a1 * m01 + m11;
		float next_m10 = -push.a2 * m00;
		float next_m11 = -push.a2 * m01;
		m00 = next_m00; m01 = next_m01; m10 = next_m10; m11 = next_m11;
		v0 = next_v0; v1 = next_v1;
	}
	uint output = task * 6;
	store_f32(storage_buffers[push.summary_index], output + 0, m00);
	store_f32(storage_buffers[push.summary_index], output + 1, m01);
	store_f32(storage_buffers[push.summary_index], output + 2, m10);
	store_f32(storage_buffers[push.summary_index], output + 3, m11);
	store_f32(storage_buffers[push.summary_index], output + 4, v0);
	store_f32(storage_buffers[push.summary_index], output + 5, v1);
}
