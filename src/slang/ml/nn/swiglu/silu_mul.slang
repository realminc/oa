// Donor: source/cpp/lib/oa/ml/shader/compute/nn/swiglu/siluMul.slang
// SiLU(first half) * second half along the final axis.

import activations;
import storage;
import attributes;

struct PushConstants {
	uint input_index;
	uint output_index;
	uint output_count;
	uint intermediate_size;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("silu_mul")]
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
	float gate = load_f32(storage_buffers[push.input_index], row_base + column);
	float up = load_f32(
		storage_buffers[push.input_index], row_base + push.intermediate_size + column);
	store_f32(storage_buffers[push.output_index], index, swiglu(gate, up));
}
