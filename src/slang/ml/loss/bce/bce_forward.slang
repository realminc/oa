// Mechanical adaptation of OA's probability-space binary cross-entropy kernel.

import storage;
import attributes;

struct PushConstants {
	uint prediction_index;
	uint target_index;
	uint output_index;
	uint element_count;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("bce")]
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
	float prediction = clamp(
		load_f32(storage_buffers[push.prediction_index], index),
		1.0e-7f,
		1.0f - 1.0e-7f
	);
	float target = load_f32(storage_buffers[push.target_index], index);
	float loss = -(target * log(prediction)
		+ (1.0f - target) * log(1.0f - prediction));
	store_f32(storage_buffers[push.output_index], index, loss);
}
