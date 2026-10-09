// Donor: source/cpp/lib/oa/ml/shader/compute/blas/groupedLinearM.slang
// Variable-row expert-major affine projection with a boundary-tile fallback.

import storage;
import attributes;

struct PushConstants {
	uint input_index;
	uint weight_index;
	uint bias_index;
	uint offsets_index;
	uint output_index;
	uint rows;
	uint output_features;
	uint input_features;
	uint experts;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

static const uint ROW_TILE = 32;
static const uint OUTPUT_TILE = 32;
static const uint REDUCTION_TILE = 16;
static const uint THREAD_ROWS = 4;
static const uint THREAD_COLUMNS = 4;
static const uint REDUCTION_PAD = REDUCTION_TILE + 1;
groupshared float shared_input[ROW_TILE * REDUCTION_PAD];
groupshared float shared_weight[OUTPUT_TILE * REDUCTION_PAD];

uint expert_for_row(uint row) {
	uint expert = 0;
	while (expert + 1 < push.experts &&
		row >= storage_buffers[push.offsets_index].Load((expert + 1) * 4)) {
		++expert;
	}
	return expert;
}

[kernel_name("grouped_linear_m")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID) {
	uint row_base = group_id.x * ROW_TILE;
	uint column_base = group_id.y * OUTPUT_TILE;
	if (row_base >= push.rows || column_base >= push.output_features) return;
	uint lane = group_thread_id.x;
	uint row_end = min(row_base + ROW_TILE, push.rows);
	uint first_expert = expert_for_row(row_base);
	uint last_expert = expert_for_row(row_end - 1);
	uint thread_row = lane / 8;
	uint thread_column = lane % 8;
	uint output_row = row_base + thread_row * THREAD_ROWS;
	uint output_column = column_base + thread_column * THREAD_COLUMNS;

	if (first_expert != last_expert) {
		for (uint i = 0; i < THREAD_ROWS; ++i) {
			uint row = output_row + i;
			if (row >= push.rows) continue;
			uint expert = expert_for_row(row);
			for (uint j = 0; j < THREAD_COLUMNS; ++j) {
				uint column = output_column + j;
				if (column >= push.output_features) continue;
				float sum = load_f32(
					storage_buffers[push.bias_index], expert * push.output_features + column);
				for (uint k = 0; k < push.input_features; ++k) {
					sum += load_f32(storage_buffers[push.input_index], row * push.input_features + k)
						* load_f32(storage_buffers[push.weight_index],
							(expert * push.output_features + column) * push.input_features + k);
				}
				store_f32(storage_buffers[push.output_index], row * push.output_features + column, sum);
			}
		}
		return;
	}

	float accumulator[THREAD_ROWS * THREAD_COLUMNS];
	[[unroll]] for (uint i = 0; i < THREAD_ROWS * THREAD_COLUMNS; ++i) accumulator[i] = 0.0;
	for (uint reduction_base = 0; reduction_base < push.input_features;
		reduction_base += REDUCTION_TILE) {
		[[unroll]] for (uint load_index = 0; load_index < 8; ++load_index) {
			uint index = lane + load_index * 64;
			uint tile_row = index / REDUCTION_TILE;
			uint tile_reduction = index % REDUCTION_TILE;
			uint row = row_base + tile_row;
			uint reduction = reduction_base + tile_reduction;
			shared_input[tile_row * REDUCTION_PAD + tile_reduction] =
				(row < push.rows && reduction < push.input_features)
				? load_f32(storage_buffers[push.input_index], row * push.input_features + reduction)
				: 0.0;
			uint column = column_base + tile_row;
			shared_weight[tile_row * REDUCTION_PAD + tile_reduction] =
				(column < push.output_features && reduction < push.input_features)
				? load_f32(storage_buffers[push.weight_index],
					(first_expert * push.output_features + column) * push.input_features + reduction)
				: 0.0;
		}
		GroupMemoryBarrierWithGroupSync();
		[[unroll]] for (uint reduction = 0; reduction < REDUCTION_TILE; ++reduction) {
			float input_values[THREAD_ROWS];
			float weight_values[THREAD_COLUMNS];
			[[unroll]] for (uint i = 0; i < THREAD_ROWS; ++i) {
				input_values[i] = shared_input[(thread_row * THREAD_ROWS + i) * REDUCTION_PAD + reduction];
			}
			[[unroll]] for (uint j = 0; j < THREAD_COLUMNS; ++j) {
				weight_values[j] = shared_weight[(thread_column * THREAD_COLUMNS + j) * REDUCTION_PAD + reduction];
			}
			[[unroll]] for (uint i = 0; i < THREAD_ROWS; ++i) {
				[[unroll]] for (uint j = 0; j < THREAD_COLUMNS; ++j) {
					accumulator[i * THREAD_COLUMNS + j] += input_values[i] * weight_values[j];
				}
			}
		}
		GroupMemoryBarrierWithGroupSync();
	}
	for (uint i = 0; i < THREAD_ROWS; ++i) {
		for (uint j = 0; j < THREAD_COLUMNS; ++j) {
			uint row = output_row + i;
			uint column = output_column + j;
			if (row < push.rows && column < push.output_features) {
				store_f32(storage_buffers[push.output_index], row * push.output_features + column,
					accumulator[i * THREAD_COLUMNS + j]
					+ load_f32(storage_buffers[push.bias_index],
						first_expert * push.output_features + column));
			}
		}
	}
}
