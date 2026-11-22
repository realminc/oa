// Deterministic Embedding scatter-add adjoint with a fused two-addend
// upstream combine. One invocation owns one table scalar and scans all
// positions; the combined output gradient `addend_a + addend_b` is
// evaluated inside the scan, so the unfused `matrix.add` buffer is never
// materialized. Every consumer of the original sum is one of these
// dispatches, which keeps the dropped buffer unobservable.

import storage;
import attributes;

struct PushConstants {
	uint indices_index;
	uint addend_a_index;
	uint addend_b_index;
	uint weight_gradient_index;
	uint index_count;
	uint num_embeddings;
	uint embedding_dim;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("embedding_backward_sum")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID) {
	uint weight_index = dispatch_thread_id.x;
	uint weight_count = push.num_embeddings * push.embedding_dim;
	if (weight_index >= weight_count) {
		return;
	}

	uint row = weight_index / push.embedding_dim;
	uint feature = weight_index % push.embedding_dim;
	float gradient = 0.0f;
	for (uint position = 0; position < push.index_count; ++position) {
		uint index = storage_buffers[push.indices_index].Load(position * 4u);
		if (index == row) {
			uint element = position * push.embedding_dim + feature;
			gradient += load_f32(storage_buffers[push.addend_a_index], element)
				+ load_f32(storage_buffers[push.addend_b_index], element);
		}
	}
	store_f32(storage_buffers[push.weight_gradient_index], weight_index, gradient);
}
