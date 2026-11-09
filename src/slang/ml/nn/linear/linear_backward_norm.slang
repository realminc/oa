// Tiled Linear data adjoint with a fused layer-norm backward prologue on the
// upstream gradient, adapted from linear_backward.slang. The linear data
// adjoint reads the layer norm's materialized input gradient
//   dx_norm[r,c] = inv_stddev[r] * (upstream[r,c] * gamma[c]
//       - sum_j upstream[r,j] * gamma[j] / output_features
//       - normalized[r,c] * sum_j upstream[r,j] * gamma[j] * normalized[r,j]
//           / output_features)
// so each workgroup first reduces both per-row statistics for its 32-row
// band into shared memory, then applies the adjoint inside the shared
// upstream tile load. The materialized dx_norm and the
// weight_contribution = upstream * normalized operand feeding the surviving
// sum_axis_pair reduction are stored only by the group_id.y == 0 column
// band so every element keeps one writer.

import storage;
import attributes;

struct PushConstants {
	uint weight_index;
	uint normalized_index;
	uint inv_stddev_index;
	uint norm_weight_index;
	uint upstream_gradient_index;
	uint norm_input_gradient_index;
	uint weight_contribution_index;
	uint input_gradient_index;
	uint batch;
	uint input_features;
	uint output_features;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

static const uint BM = 32u;
static const uint BK = 32u;
static const uint BN = 16u;
static const uint TM = 4u;
static const uint TK = 4u;
static const uint BN_PAD = BN + 1u;
groupshared float shared_output_gradient[BM * BN_PAD];
groupshared float shared_weight[BK * BN_PAD];
groupshared float2 shared_stat_partials[64];
groupshared float2 shared_row_stats[BM];

[kernel_name("linear_backward_norm")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID) {
	uint row_base = group_id.x * BM;
	uint column_base = group_id.y * BK;
	if (row_base >= push.batch || column_base >= push.input_features) {
		return;
	}

	uint thread = group_thread_id.x;
	uint thread_row = thread / 8u;
	uint thread_column = thread % 8u;
	float accumulators[TM * TK];
	[[unroll]] for (uint index = 0u; index < TM * TK; ++index) {
		accumulators[index] = 0.0f;
	}

	// Row-statistics pre-pass: the 64 threads split into 32 row pairs and
	// each lane accumulates both statistics over its half of the columns, so
	// the pair's partials combine into the full-row sums
	// (sum(upstream * gamma), sum(upstream * gamma * normalized)).
	uint stat_lane = thread / 2u;
	uint stat_row = row_base + stat_lane;
	float2 stat_partial = float2(0.0f, 0.0f);
	if (stat_row < push.batch) {
		uint row_offset = stat_row * push.output_features;
		for (uint column = thread & 1u; column < push.output_features; column += 2u) {
			float upstream = load_f32(
				storage_buffers[push.upstream_gradient_index],
				row_offset + column
			);
			float gamma = load_f32(storage_buffers[push.norm_weight_index], column);
			float scaled = upstream * gamma;
			stat_partial += float2(
				scaled,
				scaled * load_f32(storage_buffers[push.normalized_index], row_offset + column)
			);
		}
	}
	shared_stat_partials[thread] = stat_partial;
	GroupMemoryBarrierWithGroupSync();
	if ((thread & 1u) == 0u) {
		shared_row_stats[stat_lane] =
			shared_stat_partials[thread] + shared_stat_partials[thread + 1u];
	}
	GroupMemoryBarrierWithGroupSync();
	float inverse_columns = 1.0f / float(push.output_features);

	for (uint output_base = 0u; output_base < push.output_features; output_base += BN) {
		[[unroll]] for (uint load = 0u; load < 8u; ++load) {
			uint index = thread + load * 64u;
			uint output_lane = index / BN;
			uint output_feature = output_base + index % BN;
			uint row = row_base + output_lane;
			bool valid = row < push.batch && output_feature < push.output_features;
			uint element = row * push.output_features + output_feature;
			float dy = 0.0f;
			if (valid) {
				float upstream = load_f32(
					storage_buffers[push.upstream_gradient_index],
					element
				);
				float normalized = load_f32(
					storage_buffers[push.normalized_index],
					element
				);
				float gamma = load_f32(
					storage_buffers[push.norm_weight_index],
					output_feature
				);
				float2 stats = shared_row_stats[output_lane];
				dy = load_f32(storage_buffers[push.inv_stddev_index], row)
					* (upstream * gamma - stats.x * inverse_columns
						- normalized * stats.y * inverse_columns);
				if (group_id.y == 0u) {
					store_f32(storage_buffers[push.norm_input_gradient_index], element, dy);
					store_f32(
						storage_buffers[push.weight_contribution_index],
						element,
						upstream * normalized
					);
				}
			}
			shared_output_gradient[output_lane * BN_PAD + index % BN] = dy;
			uint input_feature = column_base + output_lane;
			shared_weight[output_lane * BN_PAD + index % BN] =
				input_feature < push.input_features && output_feature < push.output_features
				? load_f32(storage_buffers[push.weight_index], output_feature * push.input_features + input_feature)
				: 0.0f;
		}
		GroupMemoryBarrierWithGroupSync();

		[[unroll]] for (uint output_feature = 0u; output_feature < BN; ++output_feature) {
			float output_values[TM];
			float weight_values[TK];
			[[unroll]] for (uint row = 0u; row < TM; ++row) {
				output_values[row] = shared_output_gradient[(thread_row * TM + row) * BN_PAD + output_feature];
			}
			[[unroll]] for (uint column = 0u; column < TK; ++column) {
				weight_values[column] = shared_weight[(thread_column * TK + column) * BN_PAD + output_feature];
			}
			[[unroll]] for (uint row = 0u; row < TM; ++row) {
				[[unroll]] for (uint column = 0u; column < TK; ++column) {
					accumulators[row * TK + column] += output_values[row] * weight_values[column];
				}
			}
		}
		GroupMemoryBarrierWithGroupSync();
	}

	[[unroll]] for (uint row = 0u; row < TM; ++row) {
		[[unroll]] for (uint column = 0u; column < TK; ++column) {
			uint output_row = row_base + thread_row * TM + row;
			uint input_feature = column_base + thread_column * TK + column;
			if (output_row < push.batch && input_feature < push.input_features) {
				store_f32(
					storage_buffers[push.input_gradient_index],
					output_row * push.input_features + input_feature,
					accumulators[row * TK + column]
				);
			}
		}
	}
}
