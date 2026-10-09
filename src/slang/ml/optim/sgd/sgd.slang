// SGD-owned in-place FP32 update with optional coupled weight decay.

import storage;
import attributes;

struct PushConstants {
	uint parameter_index;
	uint gradient_index;
	uint element_count;
	float learning_rate;
	float weight_decay;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("sgd")]
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

	float parameter = load_f32(storage_buffers[push.parameter_index], index);
	float gradient = load_f32(storage_buffers[push.gradient_index], index);
	if (push.weight_decay > 0.0f) {
		gradient += push.weight_decay * parameter;
	}
	parameter -= push.learning_rate * gradient;
	store_f32(storage_buffers[push.parameter_index], index, parameter);
}
