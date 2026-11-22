// FP32 Linear forward with residual and LayerNorm epilogue. Retains the
// donor 64x64x16 tiled GEMM schedule and, when one column tile covers the
// whole output row (output_features <= BN), normalizes in-workgroup:
//   sum[m,n]    = residual[m,n] + bias[n] + sum_k input[m,k] * weight[n,k]
//   output[m,:] = layer_norm(sum[m,:]) * ln_weight + ln_bias
// Materializes every buffer the unfused `linear → add → layer_norm` triple
// would have written — the pre-residual linear output, the residual sum,
// the normalized row and the inverse stddev — so autograd saves and external
// readers observe identical storage. The matcher admits this kernel only for
// output_features <= 64; wider rows keep the linear_residual + layer_norm
// route because a row norm then spans workgroups.

import storage;
import attributes;

struct PushConstants {
	uint input_index;
	uint weight_index;
	uint bias_index;
	uint residual_index;
	uint norm_weight_index;
	uint norm_bias_index;
	uint linear_output_index;
	uint sum_index;
	uint output_index;
	uint normalized_index;
	uint inverse_stddev_index;
	uint batch;
	uint input_features;
	uint output_features;
	float epsilon;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

static const uint BM = 64u;
static const uint BN = 64u;
static const uint BK = 16u;
static const uint TM = 4u;
static const uint TN = 4u;
static const uint BK_PAD = BK + 1u;

groupshared float shared_input[BM * BK_PAD];
groupshared float shared_weight[BN * BK_PAD];
groupshared float shared_tile[BM * BN];
groupshared float row_mean[BM];
groupshared float row_inverse_stddev[BM];

void load_slice(uint row, uint column, uint buffer_index, uint row_limit, out float values[4]) {
	if (row < row_limit && column + 4u <= push.input_features) {
		float4 value = storage_buffers[buffer_index].Load<float4>(
			(row * push.input_features + column) * 4u
		);
		values[0] = value.x;
		values[1] = value.y;
		values[2] = value.z;
		values[3] = value.w;
	} else {
		[[unroll]] for (uint element = 0u; element < 4u; ++element) {
			values[element] = row < row_limit && column + element < push.input_features
				? load_f32(storage_buffers[buffer_index], row * push.input_features + column + element)
				: 0.0f;
		}
	}
}

[kernel_name("linear_residual_norm")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID) {
	uint row_base = group_id.x * BM;
	uint column_base = group_id.y * BN;
	uint thread = group_thread_id.x;
	uint thread_row = thread / 16u;
	uint thread_column = thread % 16u;
	uint local_input_row = thread_row * TM;
	uint local_weight_row = thread_column * TN;

	float accumulators[TM * TN];
	[[unroll]] for (uint index = 0u; index < TM * TN; ++index) {
		accumulators[index] = 0.0f;
	}

	uint load_row = (thread * 4u) / BK;
	uint load_column = (thread * 4u) % BK;
	uint shared_offset = load_row * BK_PAD + load_column;
	uint input_row = row_base + load_row;
	uint weight_row = column_base + load_row;
	float input_values[4];
	float weight_values[4];
	load_slice(input_row, load_column, push.input_index, push.batch, input_values);
	load_slice(weight_row, load_column, push.weight_index, push.output_features, weight_values);

	for (uint k_base = 0u; k_base < push.input_features; k_base += BK) {
		[[unroll]] for (uint element = 0u; element < 4u; ++element) {
			shared_input[shared_offset + element] = input_values[element];
			shared_weight[shared_offset + element] = weight_values[element];
		}
		GroupMemoryBarrierWithGroupSync();

		uint next_k = k_base + BK;
		if (next_k < push.input_features) {
			load_slice(input_row, next_k + load_column, push.input_index, push.batch, input_values);
			load_slice(weight_row, next_k + load_column, push.weight_index, push.output_features, weight_values);
		}

		[[unroll]] for (uint k = 0u; k < BK; ++k) {
			float input_tile[TM];
			float weight_tile[TN];
			[[unroll]] for (uint row = 0u; row < TM; ++row) {
				input_tile[row] = shared_input[(local_input_row + row) * BK_PAD + k];
			}
			[[unroll]] for (uint column = 0u; column < TN; ++column) {
				weight_tile[column] = shared_weight[(local_weight_row + column) * BK_PAD + k];
			}
			[[unroll]] for (uint row = 0u; row < TM; ++row) {
				[[unroll]] for (uint column = 0u; column < TN; ++column) {
					accumulators[row * TN + column] += input_tile[row] * weight_tile[column];
				}
			}
		}
		GroupMemoryBarrierWithGroupSync();
	}

	// Residual epilogue and row staging. Tile columns past output_features
	// stay zero so the norm reduction can run the full BN stride.
	[[unroll]] for (uint row = 0u; row < TM; ++row) {
		[[unroll]] for (uint column = 0u; column < TN; ++column) {
			uint tile_row = local_input_row + row;
			uint tile_column = local_weight_row + column;
			uint output_row = row_base + tile_row;
			uint output_column = column_base + tile_column;
			float sum = 0.0f;
			if (output_row < push.batch && output_column < push.output_features) {
				uint element = output_row * push.output_features + output_column;
				float linear_value = accumulators[row * TN + column]
					+ load_f32(storage_buffers[push.bias_index], output_column);
				sum = linear_value
					+ load_f32(storage_buffers[push.residual_index], element);
				store_f32(storage_buffers[push.linear_output_index], element, linear_value);
				store_f32(storage_buffers[push.sum_index], element, sum);
			}
			shared_tile[tile_row * BN + tile_column] = sum;
		}
	}
	GroupMemoryBarrierWithGroupSync();

	// Row statistics: one thread owns each tile row and reduces its
	// output_features serially from shared memory, so a full 64-row tile
	// costs two barriers instead of a serialized workgroup tree per row.
	// Row sums therefore accumulate in ascending column order, matching
	// the donor's numeric class without depending on barrier visibility.
	uint stat_row = thread;
	uint valid_rows = min(BM, push.batch - row_base);
	if (stat_row < valid_rows) {
		float sum = 0.0f;
		for (uint column = 0u; column < push.output_features; ++column) {
			sum += shared_tile[stat_row * BN + column];
		}
		float mean = sum / float(push.output_features);
		float square_sum = 0.0f;
		for (uint column = 0u; column < push.output_features; ++column) {
			float centered = shared_tile[stat_row * BN + column] - mean;
			square_sum += centered * centered;
		}
		float variance = square_sum / float(push.output_features);
		float inverse_stddev = rsqrt(variance + push.epsilon);
		row_mean[stat_row] = mean;
		row_inverse_stddev[stat_row] = inverse_stddev;
		store_f32(
			storage_buffers[push.inverse_stddev_index],
			row_base + stat_row,
			inverse_stddev
		);
	}
	GroupMemoryBarrierWithGroupSync();

	uint norm_row = thread / 4u;
	uint norm_lane = thread % 4u;
	uint output_row_base = row_base + norm_row;
	float row_mean_value = output_row_base < push.batch ? row_mean[norm_row] : 0.0f;
	float row_inverse_stddev_value =
		output_row_base < push.batch ? row_inverse_stddev[norm_row] : 0.0f;
	for (uint column = norm_lane; column < push.output_features; column += 4u) {
		if (output_row_base < push.batch) {
			uint element = output_row_base * push.output_features + column;
			float normalized = (shared_tile[norm_row * BN + column] - row_mean_value)
				* row_inverse_stddev_value;
			float weight = load_f32(storage_buffers[push.norm_weight_index], column);
			float bias = load_f32(storage_buffers[push.norm_bias_index], column);
			store_f32(storage_buffers[push.normalized_index], element, normalized);
			store_f32(
				storage_buffers[push.output_index],
				element,
				normalized * weight + bias
			);
		}
	}
}
