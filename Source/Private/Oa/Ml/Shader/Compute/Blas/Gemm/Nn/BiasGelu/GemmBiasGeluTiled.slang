// C[i,j] = GELU(sum_k A[i,k]*B[j,k] + bias[j])  (B transposed: C = A @ B^T + bias)
// GemmTiled with fused bias + GELU epilogue. Eliminates separate BiasAdd + Gelu dispatches.

import Gemm;
import Epilogue;

struct PushConstants {
	uint M;
	uint N;
	uint K;
};

struct FullPushConstants {
	uint a_idx;
	uint b_idx;
	uint bias_idx;
	uint out_idx;
	uint M;
	uint N;
	uint K;
};

[[vk::push_constant]] FullPushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

static const uint BM      = 64;
static const uint BN      = 64;
static const uint BK      = 16;
static const uint TM      = 4;
static const uint TN      = 4;
static const uint BK_PAD  = BK + 1;

groupshared float smA[BM * BK_PAD];
groupshared float smB[BN * BK_PAD];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
	uint tid       = lid.x;
	uint rowBase   = gid.x * BM;
	uint colBase   = gid.y * BN;
	uint threadRow = tid / 16;
	uint threadCol = tid % 16;

	float acc[TM * TN];
	for (uint i = 0; i < TM * TN; i++) acc[i] = 0.0f;

	for (uint kk = 0; kk < pc.K; kk += BK) {
		LoadTileA(heap[pc.a_idx], pc.M, pc.K, rowBase, kk, BM, BK, BK_PAD, tid, 256, smA);
		LoadTileB(heap[pc.b_idx], pc.N, pc.K, colBase, kk, BN, BK, BK_PAD, tid, 256, smB);
		GroupMemoryBarrierWithGroupSync();
		TileMatmulShared(smA, smB, acc, threadRow, threadCol, TM, TN, BK, BK_PAD);
		GroupMemoryBarrierWithGroupSync();
	}

	for (uint i = 0; i < TM; i++) {
		for (uint j = 0; j < TN; j++) {
			uint gR = rowBase + threadRow * TM + i;
			uint gC = colBase + threadCol * TN + j;
			if (gR < pc.M && gC < pc.N)
				EpilogueBiasGelu(heap[pc.out_idx], heap[pc.bias_idx], gR * pc.N + gC, gC, acc[i * TN + j]);
		}
	}
}
