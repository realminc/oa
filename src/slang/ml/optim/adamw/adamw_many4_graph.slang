// AdamW-owned replay-safe fused step for four parameter tensors.

import storage;
import attributes;

struct PushConstants {
	uint parameter0_index; uint gradient0_index; uint first_moment0_index; uint second_moment0_index;
	uint parameter1_index; uint gradient1_index; uint first_moment1_index; uint second_moment1_index;
	uint parameter2_index; uint gradient2_index; uint first_moment2_index; uint second_moment2_index;
	uint parameter3_index; uint gradient3_index; uint first_moment3_index; uint second_moment3_index;
	uint state_index;
	uint count0;
	uint count1;
	uint count2;
	uint count3;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

void update_one(
	uint index,
	uint element_count,
	uint parameter_index,
	uint gradient_index,
	uint first_moment_index,
	uint second_moment_index,
	uint step,
	float learning_rate,
	float beta1,
	float beta2,
	float epsilon,
	float weight_decay
) {
	if (index >= element_count) {
		return;
	}
	float parameter = load_f32(storage_buffers[parameter_index], index);
	float gradient = load_f32(storage_buffers[gradient_index], index);
	float first_moment = load_f32(storage_buffers[first_moment_index], index);
	float second_moment = load_f32(storage_buffers[second_moment_index], index);
	parameter -= learning_rate * weight_decay * parameter;
	float next_first_moment = beta1 * first_moment + (1.0f - beta1) * gradient;
	float next_second_moment = beta2 * second_moment + (1.0f - beta2) * gradient * gradient;
	float first_hat = next_first_moment / (1.0f - pow(beta1, float(step)));
	float second_hat = next_second_moment / (1.0f - pow(beta2, float(step)));
	parameter -= learning_rate * first_hat / (sqrt(second_hat) + epsilon);
	store_f32(storage_buffers[parameter_index], index, parameter);
	store_f32(storage_buffers[first_moment_index], index, next_first_moment);
	store_f32(storage_buffers[second_moment_index], index, next_second_moment);
}

[kernel_name("adamw_many4_graph")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID) {
	uint index = dispatch_thread_id.x;
	uint step = storage_buffers[push.state_index].Load(0u);
	float learning_rate = asfloat(storage_buffers[push.state_index].Load(4u));
	float beta1 = asfloat(storage_buffers[push.state_index].Load(8u));
	float beta2 = asfloat(storage_buffers[push.state_index].Load(12u));
	float epsilon = asfloat(storage_buffers[push.state_index].Load(16u));
	float weight_decay = asfloat(storage_buffers[push.state_index].Load(20u));
	update_one(index, push.count0, push.parameter0_index, push.gradient0_index, push.first_moment0_index, push.second_moment0_index, step, learning_rate, beta1, beta2, epsilon, weight_decay);
	update_one(index, push.count1, push.parameter1_index, push.gradient1_index, push.first_moment1_index, push.second_moment1_index, step, learning_rate, beta1, beta2, epsilon, weight_decay);
	update_one(index, push.count2, push.parameter2_index, push.gradient2_index, push.first_moment2_index, push.second_moment2_index, step, learning_rate, beta1, beta2, epsilon, weight_decay);
	update_one(index, push.count3, push.parameter3_index, push.gradient3_index, push.first_moment3_index, push.second_moment3_index, step, learning_rate, beta1, beta2, epsilon, weight_decay);
}
