// Mechanically adapted from OA C++:
// source/cpp/lib/oa/ml/shader/compute/nn/attention/mergeHeads.slang

import storage;
import attributes;

struct PushConstants {
	uint input_index;
	uint output_index;
	uint batch;
	uint sequence_length;
	uint num_heads;
	uint head_dim;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("merge_heads")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID) {
	uint output_index = dispatch_thread_id.x;
	uint count = push.batch * push.sequence_length * push.num_heads * push.head_dim;
	if (output_index >= count) {
		return;
	}
	uint feature = output_index % push.head_dim;
	uint quotient = output_index / push.head_dim;
	uint head = quotient % push.num_heads;
	quotient /= push.num_heads;
	uint sequence = quotient % push.sequence_length;
	uint batch = quotient / push.sequence_length;
	uint input_index =
		((batch * push.num_heads + head) * push.sequence_length + sequence) *
		push.head_dim + feature;
	store_f32(
		storage_buffers[push.output_index],
		output_index,
		load_f32(storage_buffers[push.input_index], input_index));
}
