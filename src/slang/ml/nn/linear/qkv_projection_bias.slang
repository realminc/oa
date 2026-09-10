// Linear-family adaptation of OA's bounded FP32 multi-output QKV projection.
// Donor: source/cpp/lib/oa/ml/shader/compute/nn/linear/qkvProjectionBias.slang

import storage;
import attributes;

struct PushConstants {
	uint input_index;
	uint query_weight_index;
	uint key_weight_index;
	uint value_weight_index;
	uint query_bias_index;
	uint key_bias_index;
	uint value_bias_index;
	uint query_output_index;
	uint key_output_index;
	uint value_output_index;
	uint m;
	uint n;
	uint k;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

static const uint BM = 32;
static const uint BN = 32;
static const uint BK = 32;
static const uint BK_PAD = BK + 1;

groupshared float shared_input[BM * BK_PAD];
groupshared float shared_query_weight[BN * BK_PAD];
groupshared float shared_key_weight[BN * BK_PAD];
groupshared float shared_value_weight[BN * BK_PAD];

[kernel_name("qkv_projection_bias")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint3 thread_id : SV_GroupThreadID) {
	uint thread = thread_id.x;
	uint row_base = group_id.x * BM;
	uint column_base = group_id.y * BN;
	uint local_row = thread / 8;
	uint local_column_base = (thread % 8) * 4;
	uint row = row_base + local_row;

	float query[4];
	float key[4];
	float value[4];
	[[unroll]] for (uint output = 0; output < 4; ++output) {
		query[output] = 0.0f;
		key[output] = 0.0f;
		value[output] = 0.0f;
	}

	for (uint k_base = 0; k_base < push.k; k_base += BK) {
		[[unroll]] for (uint load = 0; load < 4; ++load) {
			uint linear = thread + load * 256;
			uint tile_row = linear / BK;
			uint tile_k = linear % BK;
			uint global_k = k_base + tile_k;
			uint input_row = row_base + tile_row;
			uint weight_row = column_base + tile_row;
			uint shared = tile_row * BK_PAD + tile_k;
			shared_input[shared] = input_row < push.m && global_k < push.k
				? load_f32(storage_buffers[push.input_index], input_row * push.k + global_k)
				: 0.0f;
			shared_query_weight[shared] = weight_row < push.n && global_k < push.k
				? load_f32(storage_buffers[push.query_weight_index], weight_row * push.k + global_k)
				: 0.0f;
			shared_key_weight[shared] = weight_row < push.n && global_k < push.k
				? load_f32(storage_buffers[push.key_weight_index], weight_row * push.k + global_k)
				: 0.0f;
			shared_value_weight[shared] = weight_row < push.n && global_k < push.k
				? load_f32(storage_buffers[push.value_weight_index], weight_row * push.k + global_k)
				: 0.0f;
		}
		GroupMemoryBarrierWithGroupSync();

		[[unroll]] for (uint inner = 0; inner < BK; ++inner) {
			float activation = shared_input[local_row * BK_PAD + inner];
			[[unroll]] for (uint output = 0; output < 4; ++output) {
				uint local_column = local_column_base + output;
				uint weight = local_column * BK_PAD + inner;
				query[output] += activation * shared_query_weight[weight];
				key[output] += activation * shared_key_weight[weight];
				value[output] += activation * shared_value_weight[weight];
			}
		}
		GroupMemoryBarrierWithGroupSync();
	}

	if (row >= push.m) {
		return;
	}
	[[unroll]] for (uint output = 0; output < 4; ++output) {
		uint column = column_base + local_column_base + output;
		if (column >= push.n) {
			continue;
		}
		uint index = row * push.n + column;
		store_f32(
			storage_buffers[push.query_output_index],
			index,
			query[output] + load_f32(storage_buffers[push.query_bias_index], column)
		);
		store_f32(
			storage_buffers[push.key_output_index],
			index,
			key[output] + load_f32(storage_buffers[push.key_bias_index], column)
		);
		store_f32(
			storage_buffers[push.value_output_index],
			index,
			value[output] + load_f32(storage_buffers[push.value_bias_index], column)
		);
	}
}
