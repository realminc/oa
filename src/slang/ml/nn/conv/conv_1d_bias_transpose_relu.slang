// Submit-time fusion of the Conv1d bias add, NCL transpose and ReLU.
// Donor: src/slang/ml/nn/conv/conv_1d_bias_transpose.slang
// The tile load adds bias[channel] inline and the store applies
// max(0, .), so the projected [B, T, C] buffer is read once and
// the activated [B, C, T] output is written in the same pass.

import storage;
import attributes;

static const uint TILE = 32;
static const uint TILE_PAD = TILE + 1;

groupshared float tile[TILE * TILE_PAD];

struct PushConstants {
	uint input_index;
	uint bias_index;
	uint output_index;
	uint batch_size;
	uint rows;
	uint columns;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("conv_1d_bias_transpose_relu")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(TILE, TILE, 1)]
void main(uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID) {
	uint batch = group_id.z;
	uint input_row = group_id.y * TILE + group_thread_id.y;
	uint input_column = group_id.x * TILE + group_thread_id.x;
	uint input_base = batch * push.rows * push.columns;
	if (input_row < push.rows && input_column < push.columns) {
		tile[group_thread_id.y * TILE_PAD + group_thread_id.x] = load_f32(
			storage_buffers[push.input_index],
			input_base + input_row * push.columns + input_column
		) + load_f32(storage_buffers[push.bias_index], input_column);
	} else {
		tile[group_thread_id.y * TILE_PAD + group_thread_id.x] = 0.0f;
	}
	GroupMemoryBarrierWithGroupSync();

	uint output_row = group_id.x * TILE + group_thread_id.y;
	uint output_column = group_id.y * TILE + group_thread_id.x;
	if (output_row < push.columns && output_column < push.rows) {
		uint output_base = batch * push.columns * push.rows;
		store_f32(
			storage_buffers[push.output_index],
			output_base + output_row * push.rows + output_column,
			max(0.0f, tile[group_thread_id.x * TILE_PAD + group_thread_id.y])
		);
	}
}
