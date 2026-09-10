// AdamW-owned in-place FP32 update using graph-resident replay state.

import storage;
import attributes;

struct PushConstants {
	uint parameter_index;
	uint gradient_index;
	uint first_moment_index;
	uint second_moment_index;
	uint state_index;
	uint element_count;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("adamw_graph")]
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

	float step = float(storage_buffers[push.state_index].Load(0u));
	float learning_rate = asfloat(storage_buffers[push.state_index].Load(4u));
	float beta1 = asfloat(storage_buffers[push.state_index].Load(8u));
	float beta2 = asfloat(storage_buffers[push.state_index].Load(12u));
	float epsilon = asfloat(storage_buffers[push.state_index].Load(16u));
	float weight_decay = asfloat(storage_buffers[push.state_index].Load(20u));
	float parameter = load_f32(storage_buffers[push.parameter_index], index);
	float gradient = load_f32(storage_buffers[push.gradient_index], index);
	float first_moment = load_f32(storage_buffers[push.first_moment_index], index);
	float second_moment = load_f32(storage_buffers[push.second_moment_index], index);
	parameter -= learning_rate * weight_decay * parameter;
	float next_first_moment = beta1 * first_moment + (1.0f - beta1) * gradient;
	float next_second_moment = beta2 * second_moment
		+ (1.0f - beta2) * gradient * gradient;
	float first_hat = next_first_moment / (1.0f - pow(beta1, step));
	float second_hat = next_second_moment / (1.0f - pow(beta2, step));
	parameter -= learning_rate * first_hat / (sqrt(second_hat) + epsilon);

	store_f32(storage_buffers[push.parameter_index], index, parameter);
	store_f32(storage_buffers[push.first_moment_index], index, next_first_moment);
	store_f32(storage_buffers[push.second_moment_index], index, next_second_moment);
}
