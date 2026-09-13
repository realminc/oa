// Fused Mamba-3 projection preprocess adjoint.
// Donor: source/cpp/lib/oa/ml/shader/compute/nn/ssm/mamba3/mamba3PreprocessBwd.slang
// The donor's contended float atomic is adapted to a deterministic row partial.

import storage;
import ssm_math;
import attributes;

struct PushConstants {
	uint projected_index;
	uint dt_bias_index;
	uint z_gradient_index;
	uint x_gradient_index;
	uint bh_gradient_index;
	uint ch_gradient_index;
	uint dt_gradient_index;
	uint adt_gradient_index;
	uint trap_gradient_index;
	uint angle_gradient_index;
	uint projected_gradient_index;
	uint dt_bias_partial_index;
	uint rows;
	uint inner_size;
	uint state_size;
	uint heads;
	uint rope_angles;
	uint bc_rows;
	float epsilon;
	float dt_min;
	float dt_max;
	float a_floor;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

groupshared float partial[256];

float reduce_sum(float value, uint lane) {
	partial[lane] = value;
	GroupMemoryBarrierWithGroupSync();
	for (uint stride = 128u; stride > 0u; stride >>= 1u) {
		if (lane < stride) partial[lane] += partial[lane + stride];
		GroupMemoryBarrierWithGroupSync();
	}
	return partial[0];
}

float inverse_rms(uint base, uint lane) {
	float local_square_sum = 0.0f;
	for (uint index = lane; index < push.state_size; index += 256u) {
		float value = load_f32(storage_buffers[push.projected_index], base + index);
		local_square_sum += value * value;
	}
	return 1.0f / sqrt(
		reduce_sum(local_square_sum, lane) / float(push.state_size) + push.epsilon);
}

[kernel_name("mamba3_preprocess_backward")]
[domain("ml")]
[variant("fused")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID) {
	uint row = group_id.x;
	uint lane = group_thread_id.x;
	if (row >= push.rows) return;
	uint bc_width = push.bc_rows * push.state_size;
	uint b_offset = 2u * push.inner_size;
	uint c_offset = b_offset + bc_width;
	uint dd_dt_offset = c_offset + bc_width;
	uint dd_a_offset = dd_dt_offset + push.heads;
	uint trap_offset = dd_a_offset + push.heads;
	uint angle_offset = trap_offset + push.heads;
	uint projected_width = angle_offset + push.rope_angles;
	uint row_base = row * projected_width;

	for (uint group = 0u; group < push.bc_rows; ++group) {
		uint input_base = row_base + b_offset + group * push.state_size;
		uint output_base = row * bc_width + group * push.state_size;
		float inv_rms = inverse_rms(input_base, lane);
		GroupMemoryBarrierWithGroupSync();
		float local_dot = 0.0f;
		for (uint index = lane; index < push.state_size; index += 256u) {
			local_dot += load_f32(storage_buffers[push.bh_gradient_index], output_base + index)
				* load_f32(storage_buffers[push.projected_index], input_base + index);
		}
		float dot = reduce_sum(local_dot, lane);
		for (uint index = lane; index < push.state_size; index += 256u) {
			float value = load_f32(storage_buffers[push.projected_index], input_base + index);
			float gradient = load_f32(storage_buffers[push.bh_gradient_index],
				output_base + index);
			store_f32(storage_buffers[push.projected_gradient_index], input_base + index,
				ssm_rms_norm_backward(value, gradient, inv_rms, dot,
					float(push.state_size)));
		}
		GroupMemoryBarrierWithGroupSync();
	}
	for (uint group = 0u; group < push.bc_rows; ++group) {
		uint input_base = row_base + c_offset + group * push.state_size;
		uint output_base = row * bc_width + group * push.state_size;
		float inv_rms = inverse_rms(input_base, lane);
		GroupMemoryBarrierWithGroupSync();
		float local_dot = 0.0f;
		for (uint index = lane; index < push.state_size; index += 256u) {
			local_dot += load_f32(storage_buffers[push.ch_gradient_index], output_base + index)
				* load_f32(storage_buffers[push.projected_index], input_base + index);
		}
		float dot = reduce_sum(local_dot, lane);
		for (uint index = lane; index < push.state_size; index += 256u) {
			float value = load_f32(storage_buffers[push.projected_index], input_base + index);
			float gradient = load_f32(storage_buffers[push.ch_gradient_index],
				output_base + index);
			store_f32(storage_buffers[push.projected_gradient_index], input_base + index,
				ssm_rms_norm_backward(value, gradient, inv_rms, dot,
					float(push.state_size)));
		}
		GroupMemoryBarrierWithGroupSync();
	}

	for (uint index = lane; index < push.inner_size; index += 256u) {
		store_f32(storage_buffers[push.projected_gradient_index], row_base + index,
			load_f32(storage_buffers[push.z_gradient_index], row * push.inner_size + index));
		store_f32(storage_buffers[push.projected_gradient_index],
			row_base + push.inner_size + index,
			load_f32(storage_buffers[push.x_gradient_index], row * push.inner_size + index));
	}
	for (uint head = lane; head < push.heads; head += 256u) {
		uint scalar_index = row * push.heads + head;
		float raw_dt = load_f32(storage_buffers[push.projected_index],
			row_base + dd_dt_offset + head)
			+ load_f32(storage_buffers[push.dt_bias_index], head);
		float softplus = ssm_softplus(raw_dt);
		float dt = clamp(softplus, push.dt_min, push.dt_max);
		float raw_a = load_f32(storage_buffers[push.projected_index],
			row_base + dd_a_offset + head);
		float heavy_tail = ssm_heavy_tail(raw_a);
		float a_token = ssm_a_token(heavy_tail, push.a_floor);
		float dt_total_gradient =
			load_f32(storage_buffers[push.dt_gradient_index], scalar_index)
			+ load_f32(storage_buffers[push.adt_gradient_index], scalar_index) * a_token;
		float dt_mask = softplus >= push.dt_min && softplus <= push.dt_max ? 1.0f : 0.0f;
		float raw_dt_gradient = dt_total_gradient * dt_mask * (1.0f - exp(-softplus));
		store_f32(storage_buffers[push.projected_gradient_index],
			row_base + dd_dt_offset + head, raw_dt_gradient);
		store_f32(storage_buffers[push.dt_bias_partial_index], scalar_index,
			raw_dt_gradient);
		float heavy_tail_mask = heavy_tail > push.a_floor ? 1.0f : 0.0f;
		float heavy_tail_derivative = raw_a >= 0.0f
			? 1.0f
			: 1.0f / ((1.0f - raw_a) * (1.0f - raw_a));
		float raw_a_gradient =
			load_f32(storage_buffers[push.adt_gradient_index], scalar_index)
			* dt * heavy_tail_mask * -heavy_tail_derivative;
		store_f32(storage_buffers[push.projected_gradient_index],
			row_base + dd_a_offset + head, raw_a_gradient);
		store_f32(storage_buffers[push.projected_gradient_index],
			row_base + trap_offset + head,
			load_f32(storage_buffers[push.trap_gradient_index], scalar_index));
	}
	for (uint angle = lane; angle < push.rope_angles; angle += 256u) {
		store_f32(storage_buffers[push.projected_gradient_index],
			row_base + angle_offset + angle,
			load_f32(storage_buffers[push.angle_gradient_index],
				row * push.rope_angles + angle));
	}
}
