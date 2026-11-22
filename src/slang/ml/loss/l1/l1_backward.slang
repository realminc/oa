// Mechanical adaptation of OA's mean-absolute-error prediction adjoint.

import storage;
import attributes;

struct PushConstants {
	uint prediction_index;
	uint target_index;
	uint prediction_gradient_index;
	uint element_count;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("l1_backward")]
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
	float difference = load_f32(storage_buffers[push.prediction_index], index)
		- load_f32(storage_buffers[push.target_index], index);
	float inverse_count = 1.0f / float(push.element_count);
	float gradient = difference > 0.0f
		? inverse_count
		: (difference < 0.0f ? -inverse_count : 0.0f);
	store_f32(storage_buffers[push.prediction_gradient_index], index, gradient);
}
