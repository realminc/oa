// Fused grouped-Linear parameter adjoint.
// dW[e,n,k] = sum_r dY[r,n] * X[r,k]
// dB[e,n]   = sum_r dY[r,n]
//
// The weight reduction already stages every dY tile. One K-lane accumulates
// bias from those staged values, avoiding a third dispatch and a full reread of
// dY. Only the first K tile stores dB, so every [e,n] has one writer.
import Storage;

struct PushConstants {
	uint dy_idx;
	uint x_idx;
	uint offsets_idx;
	uint dw_idx;
	uint db_idx;
	uint R;
	uint N;
	uint K;
	uint E;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

static const uint BN = 32, BK = 32, BR = 16;
static const uint TN = 4, TK = 4, BR_PAD = BR + 1;
groupshared float smY[BN * BR_PAD];
groupshared float smX[BK * BR_PAD];

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
	uint nBase = gid.x * BN;
	uint kBase = gid.y * BK;
	uint e = gid.z;
	if (e >= pc.E || nBase >= pc.N || kBase >= pc.K) return;

	uint begin = heap[pc.offsets_idx].Load(e * 4);
	uint end = heap[pc.offsets_idx].Load((e + 1) * 4);
	uint tid = lid.x;
	uint threadN = tid / 8;
	uint threadK = tid % 8;
	float acc[TN * TK];
	float biasAcc[TN];
	[[unroll]] for (uint i = 0; i < TN * TK; ++i) acc[i] = 0.0;
	[[unroll]] for (uint i = 0; i < TN; ++i) biasAcc[i] = 0.0;

	for (uint rBase = begin; rBase < end; rBase += BR) {
		[[unroll]] for (uint q = 0; q < 8; ++q) {
			uint idx = tid + q * 64;
			uint oo = idx / BR;
			uint rr = idx % BR;
			uint r = rBase + rr;
			uint n = nBase + oo;
			smY[oo * BR_PAD + rr] = (n < pc.N && r < end)
				? OaLoad(heap[pc.dy_idx], r * pc.N + n) : 0.0;
			uint k = kBase + oo;
			smX[oo * BR_PAD + rr] = (k < pc.K && r < end)
				? OaLoad(heap[pc.x_idx], r * pc.K + k) : 0.0;
		}
		GroupMemoryBarrierWithGroupSync();

		[[unroll]] for (uint rr = 0; rr < BR; ++rr) {
			float yv[TN], xv[TK];
			[[unroll]] for (uint i = 0; i < TN; ++i)
				yv[i] = smY[(threadN * TN + i) * BR_PAD + rr];
			[[unroll]] for (uint j = 0; j < TK; ++j)
				xv[j] = smX[(threadK * TK + j) * BR_PAD + rr];
			[[unroll]] for (uint i = 0; i < TN; ++i) {
				if (threadK == 0 && kBase == 0) biasAcc[i] += yv[i];
				[[unroll]] for (uint j = 0; j < TK; ++j)
					acc[i * TK + j] += yv[i] * xv[j];
			}
		}
		GroupMemoryBarrierWithGroupSync();
	}

	[[unroll]] for (uint i = 0; i < TN; ++i) {
		[[unroll]] for (uint j = 0; j < TK; ++j) {
			uint n = nBase + threadN * TN + i;
			uint k = kBase + threadK * TK + j;
			if (n < pc.N && k < pc.K)
				OaStore(heap[pc.dw_idx], (e * pc.N + n) * pc.K + k,
					acc[i * TK + j]);
		}
		if (threadK == 0 && kBase == 0) {
			uint n = nBase + threadN * TN + i;
			if (n < pc.N)
				OaStore(heap[pc.db_idx], e * pc.N + n, biasAcc[i]);
		}
	}
}
