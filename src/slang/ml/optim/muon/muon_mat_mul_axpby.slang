// Muon-owned private FP32 GEMM with residual epilogue.

import storage;
import attributes;

static const uint TILE = 16;

struct PushConstants {
	uint a_index;
	uint b_index;
	uint residual_index;
	uint output_index;
	uint m;
	uint n;
	uint k;
	float alpha;
	float beta;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];
groupshared float shared_a[TILE * TILE];
groupshared float shared_b[TILE * TILE];

[kernel_name("muon_mat_mul_axpby")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(TILE, TILE, 1)]
void main(uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID) {
	uint local_row = group_thread_id.y;
	uint local_column = group_thread_id.x;
	uint row = group_id.y * TILE + local_row;
	uint column = group_id.x * TILE + local_column;
	float accumulator = 0.0f;
	for (uint tile_k = 0; tile_k < push.k; tile_k += TILE) {
		uint a_k = tile_k + local_column;
		uint b_k = tile_k + local_row;
		shared_a[local_row * TILE + local_column] = row < push.m && a_k < push.k
			? load_f32(storage_buffers[push.a_index], row * push.k + a_k) : 0.0f;
		shared_b[local_row * TILE + local_column] = b_k < push.k && column < push.n
			? load_f32(storage_buffers[push.b_index], b_k * push.n + column) : 0.0f;
		GroupMemoryBarrierWithGroupSync();
		[unroll] for (uint inner = 0; inner < TILE; ++inner) {
			accumulator += shared_a[local_row * TILE + inner]
				* shared_b[inner * TILE + local_column];
		}
		GroupMemoryBarrierWithGroupSync();
	}
	if (row < push.m && column < push.n) {
		uint index = row * push.n + column;
		float residual = load_f32(storage_buffers[push.residual_index], index);
		store_f32(storage_buffers[push.output_index], index,
			push.alpha * accumulator + push.beta * residual);
	}
}
