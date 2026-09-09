// Row-parallel parameter adjoint for narrow dense Linear layers.

import storage;
import attributes;

struct PushConstants {
	uint input_index;
	uint output_gradient_index;
	uint weight_gradient_index;
	uint bias_gradient_index;
	uint batch;
	uint input_features;
	uint output_features;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

static const uint TILE_INPUT = 32u;
static const uint REDUCTION_STRIPES = 8u;
static const uint PADDED_INPUT = TILE_INPUT + 1u;

groupshared float partial_weight[REDUCTION_STRIPES * PADDED_INPUT];
groupshared float partial_bias[REDUCTION_STRIPES];

[kernel_name("linear_parameter_backward")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(TILE_INPUT, REDUCTION_STRIPES, 1)]
void main(uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID) {
	uint output_feature = group_id.x;
	if (output_feature >= push.output_features) {
		return;
	}

	uint input_feature = group_id.y * TILE_INPUT + group_thread_id.x;
	bool valid_input = input_feature < push.input_features;
	float weight_gradient = 0.0f;
	float bias_gradient = 0.0f;
	for (uint row = group_thread_id.y; row < push.batch; row += REDUCTION_STRIPES) {
		float output_gradient = load_f32(
			storage_buffers[push.output_gradient_index],
			row * push.output_features + output_feature
		);
		if (valid_input) {
			weight_gradient += output_gradient * load_f32(
				storage_buffers[push.input_index],
				row * push.input_features + input_feature
			);
		}
		if (group_thread_id.x == 0u) {
			bias_gradient += output_gradient;
		}
	}

	uint shared_index = group_thread_id.y * PADDED_INPUT + group_thread_id.x;
	partial_weight[shared_index] = weight_gradient;
	if (group_thread_id.x == 0u) {
		partial_bias[group_thread_id.y] = bias_gradient;
	}
	GroupMemoryBarrierWithGroupSync();

	if (group_thread_id.y == 0u) {
		float weight_total = 0.0f;
		[[unroll]] for (uint stripe = 0; stripe < REDUCTION_STRIPES; ++stripe) {
			weight_total += partial_weight[stripe * PADDED_INPUT + group_thread_id.x];
		}
		if (valid_input) {
			store_f32(
				storage_buffers[push.weight_gradient_index],
				output_feature * push.input_features + input_feature,
				weight_total
			);
		}
		if (group_thread_id.x == 0u && group_id.y == 0u) {
			float bias_total = 0.0f;
			[[unroll]] for (uint stripe = 0; stripe < REDUCTION_STRIPES; ++stripe) {
				bias_total += partial_bias[stripe];
			}
			store_f32(storage_buffers[push.bias_gradient_index], output_feature, bias_total);
		}
	}
}
