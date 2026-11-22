// Grouped Linear data adjoint with a fused producer-side prologue on the
// upstream gradient, adapted from grouped_linear_m_data_backward.slang.
// Computes
//   dy[m,n] = prologue(upstream, saved)[m,n]
//   dx[m,k] = sum_n dy[m,n] * weight[expert(m),n,k]
// and also materializes dy so downstream consumers (parameter backward,
// autograd saves and observers) see exactly the buffers the unfused pair
// would have written. dy is stored only during the first segment/column
// pass so every element keeps one writer; each workgroup walks the row
// tile's expert segments and column-tile groups instead of dispatching a
// grid dimension per tile or falling back to a serial prologue pass.
// prologue_kind mirrors linear_backward_prologue.slang: 0-8 activation
// adjoints, 9 silu_mul over the packed [gate|up] projection, 10-13 the
// pointwise combines (add/sub/mul/scalar splat mul). alpha feeds
// elu/leaky_relu only.

import activations;
import storage;
import attributes;

struct PushConstants {
	uint upstream_gradient_index;
	uint saved_index;
	uint dy_index;
	uint weight_index;
	uint offsets_index;
	uint input_gradient_index;
	uint rows;
	uint output_features;
	uint input_features;
	uint experts;
	uint prologue_kind;
	float alpha;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

static const uint ROW_TILE = 32;
static const uint INPUT_TILE = 32;
static const uint REDUCTION_TILE = 16;
static const uint THREAD_ROWS = 4;
static const uint THREAD_COLUMNS = 4;
static const uint COLUMN_TILES = 2;
static const uint REDUCTION_PAD = REDUCTION_TILE + 1;
static const uint PROLOGUE_SILU_MUL = 9u;
static const uint PROLOGUE_SCALAR_MUL = 13u;
groupshared float shared_gradient[ROW_TILE * REDUCTION_PAD];
groupshared float shared_weight[COLUMN_TILES * INPUT_TILE * REDUCTION_PAD];

uint expert_for_row(uint row) {
	uint expert = 0;
	while (expert + 1 < push.experts &&
		row >= storage_buffers[push.offsets_index].Load((expert + 1) * 4)) ++expert;
	return expert;
}

float apply_prologue(float saved, float upstream) {
	switch (push.prologue_kind) {
	case 0u:
		return saved > 0.0f ? upstream : 0.0f;
	case 1u: {
		float x2 = saved * saved;
		float inner = 0.7978845608028654f * (saved + 0.044715f * x2 * saved);
		float t = tanh(inner);
		return upstream * (0.5f * (1.0f + t)
			+ 0.5f * saved * (1.0f - t * t) * 0.7978845608028654f * (1.0f + 0.134145f * x2));
	}
	case 2u:
		return upstream * silu_grad(saved);
	case 3u:
		return upstream * (1.0f - saved * saved);
	case 4u:
		return upstream * saved * (1.0f - saved);
	case 5u: {
		float softplus = saved > 20.0f ? saved : log(1.0f + exp(saved));
		float tanh_softplus = tanh(softplus);
		return upstream * (tanh_softplus
			+ saved * sigmoid(saved) * (1.0f - tanh_softplus * tanh_softplus));
	}
	case 6u:
		return upstream * (saved > 0.0f ? 1.0f : saved + push.alpha);
	case 7u:
		return saved > 0.0f ? upstream : upstream * push.alpha;
	case 8u:
		return upstream * (1.0f - exp(-saved));
	case 10u:
		return upstream + saved;
	case 11u:
		return upstream - saved;
	default:
		return upstream * saved;
	}
}

// The fused producer output evaluated at one [row, feature] element.
float prologue_value(uint row, uint feature) {
	uint element = row * push.output_features + feature;
	if (push.prologue_kind == PROLOGUE_SILU_MUL) {
		uint hidden = push.output_features / 2u;
		uint column = feature % hidden;
		uint reduced = row * hidden + column;
		float gate = load_f32(
			storage_buffers[push.saved_index],
			row * push.output_features + column
		);
		float upstream = load_f32(storage_buffers[push.upstream_gradient_index], reduced);
		if (feature < hidden) {
			float up = load_f32(
				storage_buffers[push.saved_index],
				row * push.output_features + hidden + column
			);
			return upstream * up * silu_grad(gate);
		}
		return upstream * silu(gate);
	}
	float upstream = load_f32(storage_buffers[push.upstream_gradient_index], element);
	float saved = push.prologue_kind == PROLOGUE_SCALAR_MUL
		? load_f32(storage_buffers[push.saved_index], 0)
		: load_f32(storage_buffers[push.saved_index], element);
	return apply_prologue(saved, upstream);
}

[kernel_name("grouped_backward_prologue")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID) {
	uint row_base = group_id.x * ROW_TILE;
	if (row_base >= push.rows) return;
	uint lane = group_thread_id.x;
	uint row_end = min(row_base + ROW_TILE, push.rows);
	uint first_expert = expert_for_row(row_base);
	uint last_expert = expert_for_row(row_end - 1);
	uint thread_row = lane / 8;
	uint thread_column = lane % 8;
	uint output_row = row_base + thread_row * THREAD_ROWS;

	// One workgroup owns a row tile and walks its expert segments and
	// column-tile groups: dy is produced once per (row, feature) per
	// segment, and mixed-expert tiles reuse the same staged path per
	// segment instead of recomputing the prologue for every output
	// element in a serial fallback.
	for (uint expert = first_expert; expert <= last_expert; ++expert) {
		uint segment_lo = expert == 0u
			? row_base
			: max(row_base, storage_buffers[push.offsets_index].Load(expert * 4));
		uint segment_hi = expert + 1u < push.experts
			? min(row_end, storage_buffers[push.offsets_index].Load((expert + 1u) * 4))
			: row_end;
		for (uint column_base = 0; column_base < push.input_features;
			column_base += COLUMN_TILES * INPUT_TILE) {
			float accumulator[COLUMN_TILES * THREAD_ROWS * THREAD_COLUMNS];
			[[unroll]] for (uint i = 0; i < COLUMN_TILES * THREAD_ROWS * THREAD_COLUMNS; ++i)
				accumulator[i] = 0.0;
			for (uint reduction_base = 0; reduction_base < push.output_features;
				reduction_base += REDUCTION_TILE) {
				[[unroll]] for (uint load_index = 0; load_index < 8; ++load_index) {
					uint index = lane + load_index * 64;
					uint tile_row = index / REDUCTION_TILE;
					uint tile_reduction = index % REDUCTION_TILE;
					uint row = row_base + tile_row;
					uint reduction = reduction_base + tile_reduction;
					float dy = (row < push.rows && reduction < push.output_features)
						? prologue_value(row, reduction)
						: 0.0;
					shared_gradient[tile_row * REDUCTION_PAD + tile_reduction] = dy;
					if (expert == first_expert && column_base == 0u
						&& row < push.rows && reduction < push.output_features) {
						store_f32(storage_buffers[push.dy_index],
							row * push.output_features + reduction, dy);
					}
				}
				for (uint index = lane;
					index < COLUMN_TILES * INPUT_TILE * REDUCTION_TILE; index += 64) {
					uint tile = index / (INPUT_TILE * REDUCTION_TILE);
					uint element = index % (INPUT_TILE * REDUCTION_TILE);
					uint tile_row = element / REDUCTION_TILE;
					uint tile_reduction = element % REDUCTION_TILE;
					uint k = column_base + tile * INPUT_TILE + tile_row;
					uint reduction = reduction_base + tile_reduction;
					shared_weight[tile * INPUT_TILE * REDUCTION_PAD
						+ tile_row * REDUCTION_PAD + tile_reduction] =
						(k < push.input_features && reduction < push.output_features)
						? load_f32(storage_buffers[push.weight_index],
							(expert * push.output_features + reduction)
								* push.input_features + k)
						: 0.0;
				}
				GroupMemoryBarrierWithGroupSync();
				[[unroll]] for (uint reduction = 0; reduction < REDUCTION_TILE; ++reduction) {
					float gradient_values[THREAD_ROWS];
					[[unroll]] for (uint i = 0; i < THREAD_ROWS; ++i)
						gradient_values[i] = shared_gradient[
							(thread_row * THREAD_ROWS + i) * REDUCTION_PAD + reduction];
					[[unroll]] for (uint tile = 0; tile < COLUMN_TILES; ++tile) {
						float weight_values[THREAD_COLUMNS];
						[[unroll]] for (uint j = 0; j < THREAD_COLUMNS; ++j)
							weight_values[j] = shared_weight[
								tile * INPUT_TILE * REDUCTION_PAD
								+ (thread_column * THREAD_COLUMNS + j) * REDUCTION_PAD
								+ reduction];
						[[unroll]] for (uint i = 0; i < THREAD_ROWS; ++i)
							[[unroll]] for (uint j = 0; j < THREAD_COLUMNS; ++j)
								accumulator[tile * THREAD_ROWS * THREAD_COLUMNS
									+ i * THREAD_COLUMNS + j] +=
									gradient_values[i] * weight_values[j];
					}
				}
				GroupMemoryBarrierWithGroupSync();
			}
			[[unroll]] for (uint tile = 0; tile < COLUMN_TILES; ++tile) {
				uint output_column =
					column_base + tile * INPUT_TILE + thread_column * THREAD_COLUMNS;
				for (uint i = 0; i < THREAD_ROWS; ++i) {
					for (uint j = 0; j < THREAD_COLUMNS; ++j) {
						uint row = output_row + i;
						uint k = output_column + j;
						if (row >= segment_lo && row < segment_hi
							&& k < push.input_features) {
							store_f32(storage_buffers[push.input_gradient_index],
								row * push.input_features + k,
								accumulator[tile * THREAD_ROWS * THREAD_COLUMNS
									+ i * THREAD_COLUMNS + j]);
						}
					}
				}
			}
		}
	}
}
