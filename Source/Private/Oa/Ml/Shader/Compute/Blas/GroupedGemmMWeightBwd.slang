// dW[e,n,k] = sum_r dY[r,n] * X[r,k], tiled over N/K with the expert's
// variable row range as the reduction dimension.
import Storage;
struct PushConstants { uint dy_idx; uint x_idx; uint offsets_idx; uint dw_idx; uint R; uint N; uint K; uint E; };
[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];
static const uint BN = 32, BK = 32, BR = 16, TN = 4, TK = 4, BR_PAD = BR + 1;
groupshared float smY[BN * BR_PAD];
groupshared float smX[BK * BR_PAD];
[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
	uint nBase = gid.x * BN, kBase = gid.y * BK, e = gid.z;
	if (e >= pc.E || nBase >= pc.N || kBase >= pc.K) return;
	uint begin = heap[pc.offsets_idx].Load(e * 4), end = heap[pc.offsets_idx].Load((e + 1) * 4);
	uint tid = lid.x, threadN = tid / 8, threadK = tid % 8;
	float acc[TN * TK]; [[unroll]] for (uint i = 0; i < TN * TK; ++i) acc[i] = 0.0;
	for (uint rBase = begin; rBase < end; rBase += BR) {
		[[unroll]] for (uint q = 0; q < 8; ++q) { uint idx = tid + q * 64, oo = idx / BR, rr = idx % BR; uint r = rBase + rr;
			uint n = nBase + oo; smY[oo * BR_PAD + rr] = (n < pc.N && r < end) ? OaLoad(heap[pc.dy_idx], r * pc.N + n) : 0.0;
			uint k = kBase + oo; smX[oo * BR_PAD + rr] = (k < pc.K && r < end) ? OaLoad(heap[pc.x_idx], r * pc.K + k) : 0.0; }
		GroupMemoryBarrierWithGroupSync();
		[[unroll]] for (uint rr = 0; rr < BR; ++rr) { float yv[TN], xv[TK];
			[[unroll]] for (uint i = 0; i < TN; ++i) yv[i] = smY[(threadN * TN + i) * BR_PAD + rr];
			[[unroll]] for (uint j = 0; j < TK; ++j) xv[j] = smX[(threadK * TK + j) * BR_PAD + rr];
			[[unroll]] for (uint i = 0; i < TN; ++i) [[unroll]] for (uint j = 0; j < TK; ++j) acc[i * TK + j] += yv[i] * xv[j]; }
		GroupMemoryBarrierWithGroupSync();
	}
	for (uint i = 0; i < TN; ++i) for (uint j = 0; j < TK; ++j) { uint n = nBase + threadN * TN + i, k = kBase + threadK * TK + j; if (n < pc.N && k < pc.K) OaStore(heap[pc.dw_idx], (e * pc.N + n) * pc.K + k, acc[i * TK + j]); }
}
