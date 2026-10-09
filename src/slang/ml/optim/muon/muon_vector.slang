// Muon-owned vector fallback: Nesterov momentum plus decoupled weight decay.

import storage;
import attributes;

struct PushConstants {
	uint parameter_index;
	uint gradient_index;
	uint momentum_index;
	uint element_count;
	float learning_rate;
	float beta;
	float weight_decay;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("muon_vector")]
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
	float next_momentum = push.beta * momentum + (1.0f - push.beta) * gradient;
	float update = (1.0f - push.beta) * gradient + push.beta * next_momentum;
	parameter = (1.0f - push.learning_rate * push.weight_decay) * parameter
		- push.learning_rate * update;
	store_f32(storage_buffers[push.momentum_index], index, next_momentum);
	store_f32(storage_buffers[push.parameter_index], index, parameter);
}
