// Mechanical adaptation of OA ops/gather.slang for packed U8 byte indices.
// One invocation writes one output scalar without requiring native Int8 SPIR-V.

import storage;
import attributes;

struct PushConstants {
	uint weight_index;
	uint indices_index;
	uint output_index;
	uint index_count;
	uint num_embeddings;
	uint embedding_dim;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("embedding_u8")]
[domain("ml")]
[variant("packed_u8")]
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
	uint row = load_u8(storage_buffers[push.indices_index], index_position);
	float value = row < push.num_embeddings
		? load_f32(storage_buffers[push.weight_index], row * push.embedding_dim + feature)
		: asfloat(0x7fc00000u);
	store_f32(storage_buffers[push.output_index], output_element, value);
}
