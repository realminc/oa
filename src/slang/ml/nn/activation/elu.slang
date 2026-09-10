// Mechanical Rust adaptation of OA's generated FP32 ELU kernel.
// Donor: source/cpp/lib/oa/ml/shader/compute/nn/activation/elu.slang

import storage;
import attributes;

struct PushConstants { uint input_index; uint output_index; uint element_count; float alpha; };
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("elu")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID) {
	uint index = dispatch_thread_id.x;
	if (index >= push.element_count) return;
	float input = load_f32(storage_buffers[push.input_index], index);
	float output = input > 0.0f ? input : push.alpha * (exp(input) - 1.0f);
	store_f32(storage_buffers[push.output_index], index, output);
}
