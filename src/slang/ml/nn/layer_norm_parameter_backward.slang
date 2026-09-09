// Row-parallel LayerNorm weight and bias adjoints from saved normalized values.

import storage;
import attributes;

struct PushConstants {
	uint normalized_index;
	uint output_gradient_index;
	uint weight_gradient_index;
	uint bias_gradient_index;
	uint rows;
	uint columns;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

static const uint TILE_COLUMNS = 32u;
static const uint REDUCTION_STRIPES = 8u;
static const uint PADDED_COLUMNS = TILE_COLUMNS + 1u;

groupshared float partial_weight[REDUCTION_STRIPES * PADDED_COLUMNS];
groupshared float partial_bias[REDUCTION_STRIPES * PADDED_COLUMNS];

[kernel_name("layer_norm_parameter_backward")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(TILE_COLUMNS, REDUCTION_STRIPES, 1)]
void main(uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID) {
	uint column = group_id.x * TILE_COLUMNS + group_thread_id.x;
	bool active = column < push.columns;
	float weight_gradient = 0.0f;
	float bias_gradient = 0.0f;
	if (active) {
		for (uint row = group_thread_id.y; row < push.rows; row += REDUCTION_STRIPES) {
			uint index = row * push.columns + column;
			float output_gradient = load_f32(storage_buffers[push.output_gradient_index], index);
			weight_gradient += output_gradient
				* load_f32(storage_buffers[push.normalized_index], index);
			bias_gradient += output_gradient;
		}
	}
	uint shared_index = group_thread_id.y * PADDED_COLUMNS + group_thread_id.x;
	partial_weight[shared_index] = weight_gradient;
	partial_bias[shared_index] = bias_gradient;
	GroupMemoryBarrierWithGroupSync();

	if (group_thread_id.y == 0u && active) {
		float weight_total = 0.0f;
		float bias_total = 0.0f;
		[[unroll]] for (uint stripe = 0; stripe < REDUCTION_STRIPES; ++stripe) {
			uint index = stripe * PADDED_COLUMNS + group_thread_id.x;
			weight_total += partial_weight[index];
			bias_total += partial_bias[index];
		}
		store_f32(storage_buffers[push.weight_gradient_index], column, weight_total);
		store_f32(storage_buffers[push.bias_gradient_index], column, bias_total);
	}
}
