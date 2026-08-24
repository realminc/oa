// Shared-memory FP32 BMM schedule for medium and large per-batch matrices.
// Each 16x16 workgroup computes one output tile and reuses A/B K tiles.

import storage;

struct PushConstants {
	uint a_idx;
	uint b_idx;
	uint out_idx;
	uint batch;
	uint m;
	uint k;
	uint p;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

groupshared float g_a[16][16];
groupshared float g_b[16][16];

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 group : SV_GroupID, uint3 lane : SV_GroupThreadID) {
	uint batch = group.z;
	if (batch >= pc.batch) return;
	uint row = group.y * 16 + lane.y;
	uint col = group.x * 16 + lane.x;
	float accum = 0.0;

	for (uint tile = 0; tile < pc.k; tile += 16) {
		uint aCol = tile + lane.x;
		uint bRow = tile + lane.y;
		g_a[lane.y][lane.x] = row < pc.m && aCol < pc.k
			? oaLoad(heap[pc.a_idx], (batch * pc.m + row) * pc.k + aCol)
			: 0.0;
		g_b[lane.y][lane.x] = bRow < pc.k && col < pc.p
			? oaLoad(heap[pc.b_idx], (batch * pc.k + bRow) * pc.p + col)
			: 0.0;
		GroupMemoryBarrierWithGroupSync();

		for (uint kk = 0; kk < 16; ++kk) {
			accum += g_a[lane.y][kk] * g_b[kk][lane.x];
		}
		GroupMemoryBarrierWithGroupSync();
	}

	if (row < pc.m && col < pc.p) {
		oaStore(heap[pc.out_idx], (batch * pc.m + row) * pc.p + col, accum);
	}
}
