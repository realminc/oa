// Deterministic Mamba-3 preprocess dt-bias adjoint reduction.

import storage;
import attributes;

struct PushConstants {
	uint partial_index;
	uint output_index;
	uint rows;
	uint heads;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("mamba3_preprocess_backward_reduce")]
[domain("ml")]
[variant("deterministic_reduce")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_id : SV_DispatchThreadID) {
	uint head = dispatch_id.x;
	if (head >= push.heads) return;
	float gradient = 0.0f;
	for (uint row = 0u; row < push.rows; ++row) {
		gradient += load_f32(storage_buffers[push.partial_index], row * push.heads + head);
	}
	store_f32(storage_buffers[push.output_index], head, gradient);
}
