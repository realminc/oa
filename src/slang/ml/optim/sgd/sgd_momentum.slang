// SGD-owned FP32 momentum update with donor-compatible weight decay.

import storage;
import attributes;

struct PushConstants {
	uint parameter_index;
	uint gradient_index;
	uint momentum_index;
	uint element_count;
	float learning_rate;
	float momentum_coefficient;
	float weight_decay;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("sgd_momentum")]
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
	float momentum = load_f32(storage_buffers[push.momentum_index], index);
	momentum = push.momentum_coefficient * momentum + gradient;
	store_f32(storage_buffers[push.momentum_index], index, momentum);
	if (push.weight_decay > 0.0f) {
		parameter -= push.learning_rate * momentum + push.weight_decay * parameter;
	} else {
		parameter -= push.learning_rate * momentum;
	}
	store_f32(storage_buffers[push.parameter_index], index, parameter);
}
