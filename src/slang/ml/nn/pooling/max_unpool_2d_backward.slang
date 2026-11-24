// NCHW MaxUnpool2d input adjoint: each pooled element gathers the gradient of
// the output position its recorded argmax index selected.

import storage;
import attributes;

struct PushConstants {
	uint indices_index;
	uint output_gradient_index;
	uint input_gradient_index;
	uint element_count;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("max_unpool_2d_backward")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint element_index = thread_id.x;
	if (element_index >= push.element_count) {
		return;
	}

	uint input_offset = load_u32(storage_buffers[push.indices_index], element_index);
	float gradient = load_f32(
		storage_buffers[push.output_gradient_index], input_offset
	);
	store_f32(storage_buffers[push.input_gradient_index], element_index, gradient);
}
