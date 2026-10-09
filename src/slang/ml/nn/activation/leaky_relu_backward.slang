// Mechanical Rust adaptation of OA's FP32 Leaky ReLU adjoint.
// Donor: source/cpp/lib/oa/ml/shader/compute/nn/activation/leakyReluBwd.slang

import storage;
import attributes;

struct PushConstants {
	uint input_index;
	uint output_gradient_index;
	uint input_gradient_index;
	uint element_count;
	float alpha;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("leaky_relu_backward")]
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
	float output_gradient = load_f32(storage_buffers[push.output_gradient_index], index);
	store_f32(storage_buffers[push.input_gradient_index], index,
		input > 0.0f ? output_gradient : output_gradient * push.alpha);
}
