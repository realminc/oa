// Donor: source/cpp/lib/oa/ml/shader/compute/nn/swiglu/siluMulBwd.slang

import activations;
import storage;
import attributes;

struct PushConstants {
	uint input_index;
	uint output_gradient_index;
	uint input_gradient_index;
	uint output_count;
	uint intermediate_size;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("silu_mul_backward")]
[domain("ml")]
[variant("concatenated_last_axis")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID) {
	uint index = dispatch_thread_id.x;
	if (index >= push.output_count) return;
	uint row = index / push.intermediate_size;
	uint column = index % push.intermediate_size;
	uint row_base = row * (2 * push.intermediate_size);
	uint gate_index = row_base + column;
	uint up_index = row_base + push.intermediate_size + column;
	float gate = load_f32(storage_buffers[push.input_index], gate_index);
	float up = load_f32(storage_buffers[push.input_index], up_index);
	float output_gradient = load_f32(storage_buffers[push.output_gradient_index], index);
	float sigmoid_gate = sigmoid(gate);
	float silu_gate = gate * sigmoid_gate;
	float silu_derivative = sigmoid_gate * (1.0 + gate * (1.0 - sigmoid_gate));
	store_f32(storage_buffers[push.input_gradient_index], gate_index,
		output_gradient * up * silu_derivative);
	store_f32(storage_buffers[push.input_gradient_index], up_index,
		output_gradient * silu_gate);
}
