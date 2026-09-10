import storage;
import attributes;

struct PushConstants {
	uint a_index;
	uint b_index;
	uint output_index;
	uint count;
	float gain_a;
	float gain_b;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("mix")]
[domain("audio")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint index = thread_id.x;
	if (index >= push.count) return;
	float a = load_f32(storage_buffers[push.a_index], index);
	float b = load_f32(storage_buffers[push.b_index], index);
	store_f32(storage_buffers[push.output_index], index, a * push.gain_a + b * push.gain_b);
}
