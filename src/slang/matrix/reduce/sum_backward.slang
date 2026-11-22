// Fused OARS lowering of the donor GradSum broadcast composite.

import storage;
import attributes;

struct PushConstants {
	uint output_gradient_index;
	uint input_gradient_index;
	uint outer_size;
	uint dim_size;
	uint inner_size;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("sum_backward")]
[domain("matrix")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID) {
	uint input_index = dispatch_thread_id.x;
	uint input_count = push.outer_size * push.dim_size * push.inner_size;
	if (input_index >= input_count) {
		return;
	}
	uint outer = input_index / (push.dim_size * push.inner_size);
	uint inner = input_index % push.inner_size;
	uint gradient_index = outer * push.inner_size + inner;
	store_f32(
		storage_buffers[push.input_gradient_index],
		input_index,
		load_f32(storage_buffers[push.output_gradient_index], gradient_index)
	);
}
