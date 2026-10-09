// Shared-memory FP32 BMM schedule for a left operand stored as [N,K,M].

import storage;
import attributes;

struct PushConstants {
	uint left_index;
	uint right_index;
	uint output_index;
	uint batch;
	uint rows;
	uint inner;
	uint columns;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

groupshared float left_tile[16][17];
groupshared float right_tile[16][16];

[kernel_name("bmm_tn_tiled_16")]
[domain("ml")]
[variant("tiled_16")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID) {
	uint batch_index = group_id.z;
	if (batch_index >= push.batch) {
		return;
	}
	uint row = group_id.y * 16u + group_thread_id.y;
	uint column = group_id.x * 16u + group_thread_id.x;
	float accumulator = 0.0f;
	for (uint tile = 0; tile < push.inner; tile += 16u) {
		uint left_k = tile + group_thread_id.y;
		uint left_row = group_id.y * 16u + group_thread_id.x;
		uint right_k = tile + group_thread_id.y;
		left_tile[group_thread_id.x][group_thread_id.y] = left_k < push.inner
			&& left_row < push.rows
			? load_f32(
				storage_buffers[push.left_index],
				(batch_index * push.inner + left_k) * push.rows + left_row
			)
			: 0.0f;
		right_tile[group_thread_id.y][group_thread_id.x] = right_k < push.inner
			&& column < push.columns
			? load_f32(
				storage_buffers[push.right_index],
				(batch_index * push.inner + right_k) * push.columns + column
			)
			: 0.0f;
		GroupMemoryBarrierWithGroupSync();
		for (uint k = 0; k < 16u; ++k) {
			accumulator += left_tile[group_thread_id.y][k]
				* right_tile[k][group_thread_id.x];
		}
		GroupMemoryBarrierWithGroupSync();
	}
	if (row < push.rows && column < push.columns) {
		store_f32(
			storage_buffers[push.output_index],
			(batch_index * push.rows + row) * push.columns + column,
			accumulator
		);
	}
}
