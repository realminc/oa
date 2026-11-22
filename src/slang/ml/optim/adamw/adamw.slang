// AdamW-owned in-place FP32 update over stable parameter and moment storage.

import storage;
import attributes;

struct PushConstants {
	uint parameter_index;
	uint gradient_index;
	uint first_moment_index;
	uint second_moment_index;
	uint element_count;
	uint step;
	float learning_rate;
	float beta1;
	float beta2;
	float epsilon;
	float weight_decay;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("adamw")]
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
	float first_moment = load_f32(storage_buffers[push.first_moment_index], index);
	float second_moment = load_f32(storage_buffers[push.second_moment_index], index);
	parameter -= push.learning_rate * push.weight_decay * parameter;
	float next_first_moment = push.beta1 * first_moment + (1.0f - push.beta1) * gradient;
	float next_second_moment = push.beta2 * second_moment
		+ (1.0f - push.beta2) * gradient * gradient;
	float first_hat = next_first_moment / (1.0f - pow(push.beta1, float(push.step)));
	float second_hat = next_second_moment / (1.0f - pow(push.beta2, float(push.step)));
	parameter -= push.learning_rate * first_hat / (sqrt(second_hat) + push.epsilon);

	store_f32(storage_buffers[push.parameter_index], index, parameter);
	store_f32(storage_buffers[push.first_moment_index], index, next_first_moment);
	store_f32(storage_buffers[push.second_moment_index], index, next_second_moment);
}
