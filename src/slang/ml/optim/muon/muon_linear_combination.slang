// Muon-owned private two-input linear combination.

import storage;
import attributes;

struct PushConstants {
	uint a_index;
	uint b_index;
	uint output_index;
	uint element_count;
	float scale_a;
	float scale_b;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("muon_linear_combination")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID) {
	uint index = dispatch_thread_id.x;
	if (index >= push.element_count) {
		return;
	}
	float a = load_f32(storage_buffers[push.a_index], index);
	float b = load_f32(storage_buffers[push.b_index], index);
	store_f32(storage_buffers[push.output_index], index,
		push.scale_a * a + push.scale_b * b);
}
