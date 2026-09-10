// Pointwise portion of one Elman RNN cell adjoint.

import storage;
import attributes;

struct PushConstants {
	uint gates_i_index;
	uint gates_h_index;
	uint output_gradient_index;
	uint gates_i_gradient_index;
	uint gates_h_gradient_index;
	uint element_count;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("rnn_cell_backward")]
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
	float output = tanh(
		load_f32(storage_buffers[push.gates_i_index], index)
			+ load_f32(storage_buffers[push.gates_h_index], index)
	);
	float gradient = load_f32(storage_buffers[push.output_gradient_index], index)
		* (1.0f - output * output);
	store_f32(storage_buffers[push.gates_i_gradient_index], index, gradient);
	store_f32(storage_buffers[push.gates_h_gradient_index], index, gradient);
}
