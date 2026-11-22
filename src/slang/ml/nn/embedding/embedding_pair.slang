// Paired FP32 Embedding gather plus elementwise sum, fused from the
// token/position embed + add triple. One invocation gathers one output
// scalar from each table and writes both gathered values and their sum.

import storage;
import attributes;

struct PushConstants {
	uint weight_a_index;
	uint indices_a_index;
	uint weight_b_index;
	uint indices_b_index;
	uint output_a_index;
	uint output_b_index;
	uint sum_index;
	uint index_count;
	uint num_embeddings_a;
	uint num_embeddings_b;
	uint embedding_dim;
	uint indices_a_u8;
	uint indices_b_u8;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

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

[kernel_name("embedding_pair")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID) {
	uint output_element = dispatch_thread_id.x;
	uint output_count = push.index_count * push.embedding_dim;
	if (output_element >= output_count) {
		return;
	}

	uint index_position = output_element / push.embedding_dim;
	uint feature = output_element % push.embedding_dim;
	float value_a = gather(
		push.weight_a_index,
		load_index(push.indices_a_index, index_position, push.indices_a_u8),
		push.num_embeddings_a,
		feature
	);
	float value_b = gather(
		push.weight_b_index,
		load_index(push.indices_b_index, index_position, push.indices_b_u8),
		push.num_embeddings_b,
		feature
	);
	store_f32(storage_buffers[push.output_a_index], output_element, value_a);
	store_f32(storage_buffers[push.output_b_index], output_element, value_b);
	store_f32(storage_buffers[push.sum_index], output_element, value_a + value_b);
}
