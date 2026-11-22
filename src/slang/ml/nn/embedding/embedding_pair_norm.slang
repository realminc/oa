// FP32 paired Embedding gather, elementwise sum and LayerNorm forward
// fused from the `embedding → embedding → add → layer_norm` head. One
// workgroup owns one index position: each lane gathers both table rows
// at its strided columns, writes both gathered values and their sum,
// stages the sum in shared storage, then runs the layer_norm statistics
// and affine passes. Every buffer the unfused four-node run wrote —
// both gathered outputs, the sum, the normalized row, the inverse
// stddev and the affine output — stays materialized so autograd saves
// and external readers observe identical storage. Embedding width is
// bounded by the shared row staging.

import storage;
import attributes;

static const uint MAX_COLUMNS = 1024u;

struct PushConstants {
	uint weight_a_index;
	uint indices_a_index;
	uint weight_b_index;
	uint indices_b_index;
	uint norm_weight_index;
	uint norm_bias_index;
	uint output_a_index;
	uint output_b_index;
	uint sum_index;
	uint output_index;
	uint normalized_index;
	uint inverse_stddev_index;
	uint index_count;
	uint num_embeddings_a;
	uint num_embeddings_b;
	uint embedding_dim;
	uint indices_a_u8;
	uint indices_b_u8;
	float epsilon;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

groupshared float row_values[MAX_COLUMNS];
groupshared float partial_sum[256];
groupshared float partial_square_sum[256];

uint load_index(uint buffer_index, uint position, uint packed_u8) {
	return packed_u8 != 0u
		? load_u8(storage_buffers[buffer_index], position)
		: storage_buffers[buffer_index].Load(position * 4u);
}

float gather(uint weight_index, uint row, uint num_embeddings, uint feature) {
	return row < num_embeddings
		? load_f32(storage_buffers[weight_index], row * push.embedding_dim + feature)
		: asfloat(0x7fc00000u);
}

[kernel_name("embedding_pair_norm")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID) {
	uint row = group_id.x;
	uint lane = group_thread_id.x;
	if (row >= push.index_count) {
		return;
	}

	uint index_a = load_index(push.indices_a_index, row, push.indices_a_u8);
	uint index_b = load_index(push.indices_b_index, row, push.indices_b_u8);
	uint row_offset = row * push.embedding_dim;
	for (uint column = lane; column < push.embedding_dim; column += 256u) {
		float value_a = gather(push.weight_a_index, index_a, push.num_embeddings_a, column);
		float value_b = gather(push.weight_b_index, index_b, push.num_embeddings_b, column);
		uint element = row_offset + column;
		float sum = value_a + value_b;
		store_f32(storage_buffers[push.output_a_index], element, value_a);
		store_f32(storage_buffers[push.output_b_index], element, value_b);
		store_f32(storage_buffers[push.sum_index], element, sum);
		row_values[column] = sum;
	}
	GroupMemoryBarrierWithGroupSync();

	float sum = 0.0f;
	for (uint column = lane; column < push.embedding_dim; column += 256u) {
		sum += row_values[column];
	}
	partial_sum[lane] = sum;
	GroupMemoryBarrierWithGroupSync();
	for (uint stride = 128u; stride > 0u; stride >>= 1u) {
		if (lane < stride) {
			partial_sum[lane] += partial_sum[lane + stride];
		}
		GroupMemoryBarrierWithGroupSync();
	}

	float mean = partial_sum[0] / float(push.embedding_dim);
	float square_sum = 0.0f;
	for (uint column = lane; column < push.embedding_dim; column += 256u) {
		float centered = row_values[column] - mean;
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
	float variance = partial_square_sum[0] / float(push.embedding_dim);
	float inverse_stddev = rsqrt(variance + push.epsilon);
	if (lane == 0u) {
		store_f32(storage_buffers[push.inverse_stddev_index], row, inverse_stddev);
	}
	for (uint column = lane; column < push.embedding_dim; column += 256u) {
		float normalized = (row_values[column] - mean) * inverse_stddev;
		float weight = load_f32(storage_buffers[push.norm_weight_index], column);
		float bias = load_f32(storage_buffers[push.norm_bias_index], column);
		uint element = row_offset + column;
		store_f32(storage_buffers[push.normalized_index], element, normalized);
		store_f32(
			storage_buffers[push.output_index],
			element,
			normalized * weight + bias
		);
	}
}
