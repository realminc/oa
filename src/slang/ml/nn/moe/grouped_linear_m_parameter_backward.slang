// Donor: source/cpp/lib/oa/ml/shader/compute/blas/groupedLinearMWeightBiasBwd.slang
// Fused parameter adjoint for variable-row expert-major affine projection.

import storage;
import attributes;

struct PushConstants {
	uint output_gradient_index;
	uint input_index;
	uint offsets_index;
	uint weight_gradient_index;
	uint bias_gradient_index;
	uint rows;
	uint output_features;
	uint input_features;
	uint experts;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

static const uint OUTPUT_TILE = 32;
static const uint INPUT_TILE = 32;
static const uint ROW_TILE = 16;
static const uint THREAD_OUTPUTS = 4;
static const uint THREAD_INPUTS = 4;
static const uint ROW_PAD = ROW_TILE + 1;
groupshared float shared_gradient[OUTPUT_TILE * ROW_PAD];
groupshared float shared_input[INPUT_TILE * ROW_PAD];

[kernel_name("grouped_linear_m_parameter_backward")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID) {
	uint output_base = group_id.x * OUTPUT_TILE;
	uint input_base = group_id.y * INPUT_TILE;
	uint expert = group_id.z;
	if (expert >= push.experts || output_base >= push.output_features ||
		input_base >= push.input_features) return;

	uint begin = storage_buffers[push.offsets_index].Load(expert * 4);
	uint end = storage_buffers[push.offsets_index].Load((expert + 1) * 4);
	uint lane = group_thread_id.x;
	uint thread_output = lane / 8;
	uint thread_input = lane % 8;
	float accumulator[THREAD_OUTPUTS * THREAD_INPUTS];
	float bias_accumulator[THREAD_OUTPUTS];
	[[unroll]] for (uint i = 0; i < THREAD_OUTPUTS * THREAD_INPUTS; ++i)
		accumulator[i] = 0.0;
	[[unroll]] for (uint i = 0; i < THREAD_OUTPUTS; ++i)
		bias_accumulator[i] = 0.0;

	for (uint row_base = begin; row_base < end; row_base += ROW_TILE) {
		[[unroll]] for (uint load_index = 0; load_index < 8; ++load_index) {
			uint index = lane + load_index * 64;
			uint tile_feature = index / ROW_TILE;
			uint tile_row = index % ROW_TILE;
			uint row = row_base + tile_row;
			uint output_feature = output_base + tile_feature;
			shared_gradient[tile_feature * ROW_PAD + tile_row] =
				(output_feature < push.output_features && row < end)
				? load_f32(storage_buffers[push.output_gradient_index],
					row * push.output_features + output_feature)
				: 0.0;
			uint input_feature = input_base + tile_feature;
			shared_input[tile_feature * ROW_PAD + tile_row] =
				(input_feature < push.input_features && row < end)
				? load_f32(storage_buffers[push.input_index], row * push.input_features + input_feature)
				: 0.0;
		}
		GroupMemoryBarrierWithGroupSync();

		[[unroll]] for (uint tile_row = 0; tile_row < ROW_TILE; ++tile_row) {
			float gradient_values[THREAD_OUTPUTS];
			float input_values[THREAD_INPUTS];
			[[unroll]] for (uint i = 0; i < THREAD_OUTPUTS; ++i)
				gradient_values[i] = shared_gradient[
					(thread_output * THREAD_OUTPUTS + i) * ROW_PAD + tile_row];
			[[unroll]] for (uint j = 0; j < THREAD_INPUTS; ++j)
				input_values[j] = shared_input[
					(thread_input * THREAD_INPUTS + j) * ROW_PAD + tile_row];
			[[unroll]] for (uint i = 0; i < THREAD_OUTPUTS; ++i) {
				if (thread_input == 0 && input_base == 0)
					bias_accumulator[i] += gradient_values[i];
				[[unroll]] for (uint j = 0; j < THREAD_INPUTS; ++j)
					accumulator[i * THREAD_INPUTS + j] +=
						gradient_values[i] * input_values[j];
			}
		}
		GroupMemoryBarrierWithGroupSync();
	}

	[[unroll]] for (uint i = 0; i < THREAD_OUTPUTS; ++i) {
		[[unroll]] for (uint j = 0; j < THREAD_INPUTS; ++j) {
			uint output_feature = output_base + thread_output * THREAD_OUTPUTS + i;
			uint input_feature = input_base + thread_input * THREAD_INPUTS + j;
			if (output_feature < push.output_features && input_feature < push.input_features) {
				store_f32(storage_buffers[push.weight_gradient_index],
					(expert * push.output_features + output_feature) * push.input_features
						+ input_feature,
					accumulator[i * THREAD_INPUTS + j]);
			}
		}
		if (thread_input == 0 && input_base == 0) {
			uint output_feature = output_base + thread_output * THREAD_OUTPUTS + i;
			if (output_feature < push.output_features)
				store_f32(storage_buffers[push.bias_gradient_index],
					expert * push.output_features + output_feature, bias_accumulator[i]);
		}
	}
}
