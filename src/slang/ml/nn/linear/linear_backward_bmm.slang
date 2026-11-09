// Fused BMM-into-Linear data adjoint: stage one recomputes the producer
// bmm row band and republishes it because the parameter adjoint still
// reads the intermediate dY, then stage two multiplies that band by the
// transposed weight inside the same workgroup. Both stages keep the donor
// k-tile accumulation order so the republished dY and the input gradient
// stay bit-identical to the unfused bmm + linear_backward pair.

import storage;
import attributes;

struct PushConstants {
	uint left_index;
	uint right_index;
	uint weight_index;
	uint stage1_index;
	uint input_gradient_index;
	uint batch;
	uint rows;
	uint inner;
	uint columns;
	uint input_features;
	uint kind;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

// Each workgroup owns one 16-row band across all columns; the matcher
// admits only columns and input_features of 32 or fewer so the stage-one
// tile plus the weight tile stay resident. kind 0 is the [B,R,I]x[B,I,C]
// layout of bmm_tiled_16, kind 1 the transposed [B,I,R]x[B,I,C] layout of
// bmm_tn_tiled_16.
groupshared float left_tile[16][17];
groupshared float right_tile[16][33];
groupshared float stage1_tile[16][33];
groupshared float weight_tile[32][33];

[kernel_name("linear_backward_bmm_chain")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID) {
	uint batch_index = group_id.z;
	if (batch_index >= push.batch) {
		return;
	}
	uint row = group_id.y * 16u + group_thread_id.y;
	uint right_row = group_thread_id.y;
	uint first_column = group_thread_id.x;
	uint second_column = group_thread_id.x + 16u;

	float first = 0.0f;
	float second = 0.0f;
	for (uint tile = 0u; tile < push.inner; tile += 16u) {
		uint left_k = tile + group_thread_id.x;
		left_tile[group_thread_id.y][group_thread_id.x] =
			row < push.rows && left_k < push.inner
			? load_f32(
				storage_buffers[push.left_index],
				push.kind == 0u
					? (batch_index * push.rows + row) * push.inner + left_k
					: (batch_index * push.inner + left_k) * push.rows + row
			)
			: 0.0f;
		uint right_k = tile + right_row;
		bool right_valid = right_k < push.inner;
		right_tile[right_row][first_column] =
			right_valid && first_column < push.columns
			? load_f32(
				storage_buffers[push.right_index],
				(batch_index * push.inner + right_k) * push.columns + first_column
			)
			: 0.0f;
		right_tile[right_row][second_column] =
			right_valid && second_column < push.columns
			? load_f32(
				storage_buffers[push.right_index],
				(batch_index * push.inner + right_k) * push.columns + second_column
			)
			: 0.0f;
		GroupMemoryBarrierWithGroupSync();
		for (uint k = 0u; k < 16u; ++k) {
			float left_value = left_tile[group_thread_id.y][k];
			first += left_value * right_tile[k][first_column];
			second += left_value * right_tile[k][second_column];
		}
		GroupMemoryBarrierWithGroupSync();
	}

	// Publish dY: the parameter adjoint consumes this buffer, so the fused
	// node must materialize the same values the producer bmm wrote.
	stage1_tile[group_thread_id.y][first_column] = first;
	stage1_tile[group_thread_id.y][second_column] = second;
	if (row < push.rows) {
		uint stage1_base = (batch_index * push.rows + row) * push.columns;
		if (first_column < push.columns) {
			store_f32(storage_buffers[push.stage1_index], stage1_base + first_column, first);
		}
		if (second_column < push.columns) {
			store_f32(storage_buffers[push.stage1_index], stage1_base + second_column, second);
		}
	}

	uint thread = group_thread_id.y * 16u + group_thread_id.x;
	for (uint element = thread; element < 1024u; element += 256u) {
		uint weight_row = element / 32u;
		uint weight_column = element % 32u;
		weight_tile[weight_row][weight_column] =
			weight_row < push.columns && weight_column < push.input_features
			? load_f32(
				storage_buffers[push.weight_index],
				weight_row * push.input_features + weight_column
			)
			: 0.0f;
	}
	GroupMemoryBarrierWithGroupSync();

	// Stage two: input_gradient[row, c] = sum_f dY[row, f] * weight[f, c]
	// accumulated sequentially over output features like the donor.
	float input_first = 0.0f;
	float input_second = 0.0f;
	for (uint feature = 0u; feature < push.columns; ++feature) {
		float stage1_value = stage1_tile[group_thread_id.y][feature];
		input_first += stage1_value * weight_tile[feature][first_column];
		input_second += stage1_value * weight_tile[feature][second_column];
	}
	if (row < push.rows) {
		uint gradient_base = (batch_index * push.rows + row) * push.input_features;
		if (first_column < push.input_features) {
			store_f32(storage_buffers[push.input_gradient_index], gradient_base + first_column, input_first);
		}
		if (second_column < push.input_features) {
			store_f32(storage_buffers[push.input_gradient_index], gradient_base + second_column, input_second);
		}
	}
}
