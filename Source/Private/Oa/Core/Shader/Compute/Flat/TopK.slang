// TopK — top-k values and indices along the last dimension.
// One workgroup per row; each does K cooperative-argmax passes, excluding the
// indices already selected in earlier passes. Ties break toward the lower index
// for determinism. Output layout: [row, k].
//
// Buffer indices (x, vals, idxs) are auto-prepended to the push constants by the
// bindless dispatch convention, so the host supplies only {T, E, K}.
// Requests beyond kMaxK are rejected; there is no hidden CPU fallback.

import Storage;

struct PushConstants {
	uint x_idx;     // input  [T, E] (row-major)
	uint vals_idx;  // output values  [T, K]
	uint idxs_idx;  // output indices [T, K] (int32)
	uint T;         // number of rows
	uint E;         // elements per row
	uint K;         // top-k
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

static const uint THREADS = 256;
static const uint kMaxK   = 1024; // groupshared bound; host guards K <= kMaxK

groupshared float sval[THREADS];
groupshared uint  sidx[THREADS];
groupshared uint  taken[kMaxK];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
	uint row = gid.x;
	if (row >= pc.T) return;
	uint lid  = tid.x;
	uint base = row * pc.E;

	for (uint k = 0; k < pc.K; ++k) {
		float bestVal = -3.402823466e+38f;
		uint  bestIdx = 0xFFFFFFFFu;

		for (uint i = lid; i < pc.E; i += THREADS) {
			bool isTaken = false;
			for (uint t = 0; t < k; ++t) {
				if (taken[t] == i) { isTaken = true; break; }
			}
			if (isTaken) continue;
			float v = OaLoad(heap[pc.x_idx], base + i);
			if (v > bestVal || (v == bestVal && i < bestIdx)) {
				bestVal = v;
				bestIdx = i;
			}
		}

		sval[lid] = bestVal;
		sidx[lid] = bestIdx;
		GroupMemoryBarrierWithGroupSync();

		for (uint s = 128; s > 0; s >>= 1) {
			if (lid < s) {
				if (sval[lid + s] > sval[lid] ||
				    (sval[lid + s] == sval[lid] && sidx[lid + s] < sidx[lid])) {
					sval[lid] = sval[lid + s];
					sidx[lid] = sidx[lid + s];
				}
			}
			GroupMemoryBarrierWithGroupSync();
		}

		if (lid == 0) {
			taken[k] = sidx[0];
			OaStore(heap[pc.vals_idx], row * pc.K + k, sval[0]);
			heap[pc.idxs_idx].Store((row * pc.K + k) * 4, sidx[0]);
		}
		GroupMemoryBarrierWithGroupSync();
	}
}
