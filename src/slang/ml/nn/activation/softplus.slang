// Mechanical Rust adaptation of OA's overflow-safe FP32 Softplus kernel.
// Donor: source/cpp/lib/oa/ml/shader/compute/nn/activation/softplus.slang

import storage;
import attributes;

struct PushConstants { uint input_index; uint output_index; uint element_count; };
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("softplus")]
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
	float output = max(input, 0.0f) + log(1.0f + exp(-abs(input)));
	store_f32(storage_buffers[push.output_index], index, output);
}
