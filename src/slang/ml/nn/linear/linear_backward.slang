// Tiled Linear data adjoint adapted from OA's linearDataBwd.slang.

import storage;
import attributes;

struct PushConstants {
	uint weight_index;
	uint output_gradient_index;
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

[kernel_name("linear_backward")]
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

	for (uint output_base = 0u; output_base < push.output_features; output_base += BN) {
		[[unroll]] for (uint load = 0u; load < 8u; ++load) {
			uint index = thread + load * 64u;
			uint output_lane = index / BN;
			uint output_feature = output_base + index % BN;
			uint row = row_base + output_lane;
			shared_output_gradient[output_lane * BN_PAD + index % BN] =
				row < push.batch && output_feature < push.output_features
				? load_f32(storage_buffers[push.output_gradient_index], row * push.output_features + output_feature)
				: 0.0f;
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
