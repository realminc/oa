// Stable dropless expert-major route packing. Input route order is token-major
// [t,k]. A workgroup-defined parallel prefix scan ranks matching routes in
// each 256-route chunk. This preserves deterministic route order without the
// old O(THREADS^2) construction or any subgroup-layout assumptions.

struct PushConstants {
	uint indices_idx;
	uint counts_idx;
	uint offsets_idx;
	uint packedToken_idx;
	uint packedExpert_idx;
	uint packedSlot_idx;
	uint inverse_idx;
	uint T;
	uint K;
	uint E;
	uint routes;
};
[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

static const uint THREADS = 256;
groupshared uint counts[THREADS];
groupshared uint cursors[THREADS];
groupshared uint scanA[THREADS];
groupshared uint scanB[THREADS];
groupshared uint expertBase;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_GroupThreadID) {
	uint lane = tid.x;
	if (lane < pc.E) counts[lane] = 0;
	GroupMemoryBarrierWithGroupSync();

	for (uint route = lane; route < pc.routes; route += THREADS) {
		uint expert = heap[pc.indices_idx].Load(route * 4);
		if (expert < pc.E) {
			uint ignored;
			InterlockedAdd(counts[expert], 1, ignored);
		}
	}
	GroupMemoryBarrierWithGroupSync();

	if (lane == 0) {
		uint offset = 0;
		for (uint expert = 0; expert < pc.E; ++expert) {
			uint count = counts[expert];
			heap[pc.counts_idx].Store(expert * 4, count);
			heap[pc.offsets_idx].Store(expert * 4, offset);
			cursors[expert] = offset;
			offset += count;
		}
		heap[pc.offsets_idx].Store(pc.E * 4, offset);
	}
	GroupMemoryBarrierWithGroupSync();

	for (uint base = 0; base < pc.routes; base += THREADS) {
		uint route = base + lane;
		uint routeExpert = route < pc.routes
			? heap[pc.indices_idx].Load(route * 4) : pc.E;
		for (uint expert = 0; expert < pc.E; ++expert) {
			uint selected = routeExpert == expert ? 1 : 0;
			scanA[lane] = selected;
			GroupMemoryBarrierWithGroupSync();

			bool inA = true;
			for (uint offset = 1; offset < THREADS; offset <<= 1) {
				uint value = inA ? scanA[lane] : scanB[lane];
				if (lane >= offset)
					value += inA ? scanA[lane - offset] : scanB[lane - offset];
				if (inA) scanB[lane] = value;
				else scanA[lane] = value;
				GroupMemoryBarrierWithGroupSync();
				inA = !inA;
			}
			uint inclusiveRank = inA ? scanA[lane] : scanB[lane];
			uint chunkCount = inA ? scanA[THREADS - 1] : scanB[THREADS - 1];
			if (lane == 0) {
				expertBase = cursors[expert];
				cursors[expert] += chunkCount;
			}
			GroupMemoryBarrierWithGroupSync();

			if (selected != 0) {
				uint packed = expertBase + inclusiveRank - 1;
				heap[pc.packedToken_idx].Store(packed * 4, route / pc.K);
				heap[pc.packedExpert_idx].Store(packed * 4, expert);
				heap[pc.packedSlot_idx].Store(packed * 4, route);
				heap[pc.inverse_idx].Store(route * 4, packed);
			}
		}
	}
}
