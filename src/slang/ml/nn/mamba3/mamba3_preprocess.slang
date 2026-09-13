// Fused Mamba-3 projection split, B/C RMS normalization, and discretization.
// Donor: source/cpp/lib/oa/ml/shader/compute/nn/ssm/mamba3/mamba3Preprocess.slang

import storage;
import ssm_math;
import attributes;

struct PushConstants {
	uint projected_index;
	uint dt_bias_index;
	uint z_index;
	uint x_index;
	uint bh_index;
	uint ch_index;
	uint dt_index;
	uint adt_index;
	uint trap_index;
	uint angle_index;
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

float inverse_rms(uint base, uint lane) {
	float local_square_sum = 0.0f;
	for (uint index = lane; index < push.state_size; index += 256u) {
		float value = load_f32(storage_buffers[push.projected_index], base + index);
		local_square_sum += value * value;
	}
	partial[lane] = local_square_sum;
	GroupMemoryBarrierWithGroupSync();
	for (uint stride = 128u; stride > 0u; stride >>= 1u) {
		if (lane < stride) partial[lane] += partial[lane + stride];
		GroupMemoryBarrierWithGroupSync();
	}
	return 1.0f / sqrt(partial[0] / float(push.state_size) + push.epsilon);
}

[kernel_name("mamba3_preprocess")]
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
		for (uint index = lane; index < push.state_size; index += 256u) {
			float value = load_f32(storage_buffers[push.projected_index], input_base + index);
			store_f32(storage_buffers[push.bh_index], output_base + index,
				ssm_rms_norm(value, inv_rms));
		}
		GroupMemoryBarrierWithGroupSync();
	}
	for (uint group = 0u; group < push.bc_rows; ++group) {
		uint input_base = row_base + c_offset + group * push.state_size;
		uint output_base = row * bc_width + group * push.state_size;
		float inv_rms = inverse_rms(input_base, lane);
		for (uint index = lane; index < push.state_size; index += 256u) {
			float value = load_f32(storage_buffers[push.projected_index], input_base + index);
			store_f32(storage_buffers[push.ch_index], output_base + index,
				ssm_rms_norm(value, inv_rms));
		}
		GroupMemoryBarrierWithGroupSync();
	}

	for (uint index = lane; index < push.inner_size; index += 256u) {
		store_f32(storage_buffers[push.z_index], row * push.inner_size + index,
			load_f32(storage_buffers[push.projected_index], row_base + index));
		store_f32(storage_buffers[push.x_index], row * push.inner_size + index,
			load_f32(storage_buffers[push.projected_index],
				row_base + push.inner_size + index));
	}
	for (uint head = lane; head < push.heads; head += 256u) {
		uint scalar_index = row * push.heads + head;
		float raw_dt = load_f32(storage_buffers[push.projected_index],
			row_base + dd_dt_offset + head)
			+ load_f32(storage_buffers[push.dt_bias_index], head);
		float dt = ssm_softplus_clamp(raw_dt, push.dt_min, push.dt_max);
		float raw_a = load_f32(storage_buffers[push.projected_index],
			row_base + dd_a_offset + head);
		store_f32(storage_buffers[push.dt_index], scalar_index, dt);
		store_f32(storage_buffers[push.adt_index], scalar_index,
			ssm_adt(raw_a, dt, push.a_floor));
		store_f32(storage_buffers[push.trap_index], scalar_index,
			load_f32(storage_buffers[push.projected_index],
				row_base + trap_offset + head));
	}
	for (uint angle = lane; angle < push.rope_angles; angle += 256u) {
		store_f32(storage_buffers[push.angle_index], row * push.rope_angles + angle,
			load_f32(storage_buffers[push.projected_index],
				row_base + angle_offset + angle));
	}
}
