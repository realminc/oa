// FP32 Linear forward adapted from OA's 64x64x16 tiled GEMM fallback kernel.
// C[m,n] = sum_k input[m,k] * weight[n,k] + bias[n].

import storage;
import attributes;

struct PushConstants {
	uint input_index;
	uint weight_index;
	uint bias_index;
	uint output_index;
	uint batch;
	uint input_features;
	uint output_features;
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

[kernel_name("linear")]
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

	[[unroll]] for (uint row = 0u; row < TM; ++row) {
		[[unroll]] for (uint column = 0u; column < TN; ++column) {
			uint output_row = row_base + local_input_row + row;
			uint output_column = column_base + local_weight_row + column;
			if (output_row < push.batch && output_column < push.output_features) {
				store_f32(
					storage_buffers[push.output_index],
					output_row * push.output_features + output_column,
					accumulators[row * TN + column]
						+ load_f32(storage_buffers[push.bias_index], output_column)
				);
			}
		}
	}
}
