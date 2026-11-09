// Submit-time fusion of the Conv1d bias add and a cancelled transpose
// pair — the NCL transpose followed by its exact inverse reduces to a
// biased elementwise copy of the projected [B, T, C] buffer.
// Donor: src/slang/ml/nn/conv/conv_1d_bias_add.slang

import storage;
import attributes;

struct PushConstants {
	uint input_index;
	uint bias_index;
	uint output_index;
	uint rows;
	uint columns;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("conv_1d_bias_copy")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint index = thread_id.x;
	if (index >= push.rows * push.columns) {
		return;
	}
	uint column = index % push.columns;
	store_f32(
		storage_buffers[push.output_index],
		index,
		load_f32(storage_buffers[push.input_index], index)
			+ load_f32(storage_buffers[push.bias_index], column)
	);
}
