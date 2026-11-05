// Mechanical adaptation of OA's full Max reduction adjoint: each element equal
// to the forward maximum receives the upstream scalar gradient, others zero.

import storage;
import attributes;

struct PushConstants {
	uint input_index;
	uint maximum_index;
	uint output_gradient_index;
	uint input_gradient_index;
	uint element_count;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("max_backward")]
[domain("matrix")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint index = thread_id.x;
	if (index >= push.element_count) return;

	float value = load_f32(storage_buffers[push.input_index], index);
	float maximum = load_f32(storage_buffers[push.maximum_index], 0);
	float gradient = load_f32(storage_buffers[push.output_gradient_index], 0);

	store_f32(storage_buffers[push.input_gradient_index], index, (value == maximum) ? gradient : 0.0f);
}
