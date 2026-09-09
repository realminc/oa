// FP32 LayerNorm over the last dimension. One workgroup owns one row.

import storage;
import attributes;

struct PushConstants {
	uint input_index;
	uint weight_index;
	uint bias_index;
	uint output_index;
	uint normalized_index;
	uint inverse_stddev_index;
	uint rows;
	uint columns;
	float epsilon;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

groupshared float partial_sum[256];
groupshared float partial_square_sum[256];

[kernel_name("layer_norm")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID) {
	uint row = group_id.x;
	uint lane = group_thread_id.x;
	if (row >= push.rows) {
		return;
	}

	float sum = 0.0f;
	uint row_offset = row * push.columns;
	for (uint column = lane; column < push.columns; column += 256u) {
		float value = load_f32(storage_buffers[push.input_index], row_offset + column);
		sum += value;
	}
	partial_sum[lane] = sum;
	GroupMemoryBarrierWithGroupSync();

	for (uint stride = 128u; stride > 0u; stride >>= 1u) {
		if (lane < stride) {
			partial_sum[lane] += partial_sum[lane + stride];
		}
		GroupMemoryBarrierWithGroupSync();
	}

	float mean = partial_sum[0] / float(push.columns);
	float square_sum = 0.0f;
	for (uint column = lane; column < push.columns; column += 256u) {
		float centered = load_f32(storage_buffers[push.input_index], row_offset + column) - mean;
		square_sum += centered * centered;
	}
	partial_square_sum[lane] = square_sum;
	GroupMemoryBarrierWithGroupSync();
	for (uint stride = 128u; stride > 0u; stride >>= 1u) {
		if (lane < stride) {
			partial_square_sum[lane] += partial_square_sum[lane + stride];
		}
		GroupMemoryBarrierWithGroupSync();
	}
	float variance = partial_square_sum[0] / float(push.columns);
	float inverse_stddev = rsqrt(variance + push.epsilon);
	if (lane == 0u) {
		store_f32(storage_buffers[push.inverse_stddev_index], row, inverse_stddev);
	}
	for (uint column = lane; column < push.columns; column += 256u) {
		float value = load_f32(storage_buffers[push.input_index], row_offset + column);
		float weight = load_f32(storage_buffers[push.weight_index], column);
		float bias = load_f32(storage_buffers[push.bias_index], column);
		float normalized = (value - mean) * inverse_stddev;
		store_f32(storage_buffers[push.normalized_index], row_offset + column, normalized);
		store_f32(
			storage_buffers[push.output_index],
			row_offset + column,
			normalized * weight + bias
		);
	}
}
