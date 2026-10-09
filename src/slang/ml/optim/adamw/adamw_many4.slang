// AdamW-owned fused step for four parameter tensors.

import storage;
import attributes;

struct PushConstants {
	uint parameter0_index; uint gradient0_index; uint first_moment0_index; uint second_moment0_index;
	uint parameter1_index; uint gradient1_index; uint first_moment1_index; uint second_moment1_index;
	uint parameter2_index; uint gradient2_index; uint first_moment2_index; uint second_moment2_index;
	uint parameter3_index; uint gradient3_index; uint first_moment3_index; uint second_moment3_index;
	uint count0;
	uint count1;
	uint count2;
	uint count3;
	float learning_rate;
	float beta1;
	float beta2;
	float epsilon;
	float weight_decay;
	uint step;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

void update_one(
	uint index,
	uint element_count,
	uint parameter_index,
	uint gradient_index,
	uint first_moment_index,
	uint second_moment_index
) {
	if (index >= element_count) {
		return;
	}
	float parameter = load_f32(storage_buffers[parameter_index], index);
	float gradient = load_f32(storage_buffers[gradient_index], index);
	float first_moment = load_f32(storage_buffers[first_moment_index], index);
	float second_moment = load_f32(storage_buffers[second_moment_index], index);
	parameter -= push.learning_rate * push.weight_decay * parameter;
	float next_first_moment = push.beta1 * first_moment + (1.0f - push.beta1) * gradient;
	float next_second_moment = push.beta2 * second_moment
		+ (1.0f - push.beta2) * gradient * gradient;
	float first_hat = next_first_moment / (1.0f - pow(push.beta1, float(push.step)));
	float second_hat = next_second_moment / (1.0f - pow(push.beta2, float(push.step)));
	parameter -= push.learning_rate * first_hat / (sqrt(second_hat) + push.epsilon);
	store_f32(storage_buffers[parameter_index], index, parameter);
	store_f32(storage_buffers[first_moment_index], index, next_first_moment);
	store_f32(storage_buffers[second_moment_index], index, next_second_moment);
}

[kernel_name("adamw_many4")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID) {
	uint index = dispatch_thread_id.x;
	update_one(index, push.count0, push.parameter0_index, push.gradient0_index, push.first_moment0_index, push.second_moment0_index);
	update_one(index, push.count1, push.parameter1_index, push.gradient1_index, push.first_moment1_index, push.second_moment1_index);
	update_one(index, push.count2, push.parameter2_index, push.gradient2_index, push.first_moment2_index, push.second_moment2_index);
	update_one(index, push.count3, push.parameter3_index, push.gradient3_index, push.first_moment3_index, push.second_moment3_index);
}
