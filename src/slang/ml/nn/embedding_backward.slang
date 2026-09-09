// Deterministic scatter-add adjoint. One invocation owns one table scalar.

import storage;
import attributes;

struct PushConstants {
	uint indices_index;
	uint output_gradient_index;
	uint weight_gradient_index;
	uint index_count;
	uint num_embeddings;
	uint embedding_dim;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("embedding_backward")]
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
			gradient += load_f32(
				storage_buffers[push.output_gradient_index],
				position * push.embedding_dim + feature
			);
		}
	}
	store_f32(storage_buffers[push.weight_gradient_index], weight_index, gradient);
}
