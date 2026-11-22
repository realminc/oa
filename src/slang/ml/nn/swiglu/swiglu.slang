// Two-input SwiGLU operation adapted from OA's donor kernel.
// Donor: source/cpp/lib/oa/ml/shader/compute/nn/swiglu/swiglu.slang

import activations;
import storage;
import attributes;

struct PushConstants {
	uint gate_index;
	uint up_index;
	uint output_index;
	uint element_count;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("swiglu")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID) {
	uint index = dispatch_thread_id.x;
	if (index >= push.element_count) return;
	float gate = load_f32(storage_buffers[push.gate_index], index);
	float up = load_f32(storage_buffers[push.up_index], index);
	store_f32(storage_buffers[push.output_index], index, swiglu(gate, up));
}
