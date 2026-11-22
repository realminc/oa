// Mechanical adaptation of OA's output-saved Softmax adjoint.

import storage;
import attributes;

struct PushConstants {
	uint forward_output_index;
	uint output_gradient_index;
	uint input_gradient_index;
	uint outer_size;
	uint dim_size;
	uint inner_size;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

groupshared float shared_values[256];

[kernel_name("softmax_backward")]
[domain("matrix")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID) {
	uint group = group_id.x;
	uint lane = group_thread_id.x;
	if (group >= push.outer_size * push.inner_size) {
		return;
	}

	uint outer = group / push.inner_size;
	uint inner = group % push.inner_size;
	uint base = outer * push.dim_size * push.inner_size + inner;

	float local_dot = 0.0f;
	for (uint index = lane; index < push.dim_size; index += 256) {
		uint element = base + index * push.inner_size;
		local_dot += load_f32(storage_buffers[push.output_gradient_index], element)
			* load_f32(storage_buffers[push.forward_output_index], element);
	}
	shared_values[lane] = local_dot;
	GroupMemoryBarrierWithGroupSync();
	for (uint stride = 128; stride > 0; stride >>= 1) {
		if (lane < stride) {
			shared_values[lane] += shared_values[lane + stride];
		}
		GroupMemoryBarrierWithGroupSync();
	}
	float dot = shared_values[0];
	GroupMemoryBarrierWithGroupSync();

	for (uint index = lane; index < push.dim_size; index += 256) {
		uint element = base + index * push.inner_size;
		float value = load_f32(storage_buffers[push.forward_output_index], element);
		float gradient = load_f32(storage_buffers[push.output_gradient_index], element);
		store_f32(
			storage_buffers[push.input_gradient_index],
			element,
			value * (gradient - dot)
		);
	}
}
