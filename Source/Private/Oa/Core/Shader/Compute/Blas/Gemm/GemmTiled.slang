// C[i,j] = sum_k A[i,k] * B[j,k]  (B is transposed: C = A @ B^T)
// Tiled shared-memory matmul — 64x64 output tiles, 16-wide K tiles.
// 256 threads per workgroup. Each thread computes a 4x4 sub-tile.
//
// This is the fp32/bf16 scalar-fallback GEMM — the workhorse on hardware without a
// usable KHR cooperative-matrix path (e.g. Intel Xe iGPUs). It is SOFTWARE-PIPELINED
// (register-prefetch double buffering): each K-iteration issues the NEXT tile's global
// loads into registers *before* running the current tile's FMAs, so global-load latency
// overlaps compute instead of stalling the workgroup at every barrier. That is the
// biggest fp32-tiled win per OaGemmRewritePlan.md §1 and mirrors the reference
// shmem.comp pipeline. SMEM staging uses 128-bit float4 vectorized loads (fp32) with a
// scalar OaLoad fallback for bf16 (DTYPE=1) and out-of-bounds tails.
//
// Numerics are identical to the un-pipelined version (same accumulation order); only
// the load/compute scheduling changed. Validate with TutorialCoreMatMulIntro
// (norm_err must stay 0) before trusting the throughput numbers.

import Storage;

struct FullPushConstants {
	uint a_idx;
	uint b_idx;
	uint out_idx;
	uint M;
	uint N;
	uint K;
};

[[vk::push_constant]] FullPushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

static const uint BM = 64;
static const uint BN = 64;
static const uint BK = 16;
static const uint TM = 4;
static const uint TN = 4;

// 256 threads arranged as 16x16 grid, each computes 4x4 output elements
static const uint THREADS_ROW = 16;
static const uint THREADS_COL = 16;

// +1 padding avoids bank conflicts when threads read different rows at same k
static const uint BK_PAD = BK + 1;
groupshared float smA[BM * BK_PAD];
groupshared float smB[BN * BK_PAD];

// Load this thread's 4-element K-slice of a tile (matrix row gRow, K-columns
// gCol0..gCol0+3) into reg[4]. fp32 uses one 128-bit float4 load; bf16 (DTYPE=1) and
// out-of-bounds tails fall back to scalar OaLoad — bit-identical to the original loop.
void LoadSlice(uint gRow, uint gCol0, uint bufIdx, uint dimLimit, out float reg[4]) {
	if (!OaIsBf16() && gRow < dimLimit && gCol0 + 4u <= pc.K) {
		float4 v = heap[bufIdx].Load<float4>((gRow * pc.K + gCol0) * 4u);
		reg[0] = v.x; reg[1] = v.y; reg[2] = v.z; reg[3] = v.w;
	} else {
		[[unroll]] for (uint e = 0; e < 4u; ++e) {
			reg[e] = (gRow < dimLimit && gCol0 + e < pc.K)
				? OaLoad(heap[bufIdx], gRow * pc.K + gCol0 + e)
				: 0.0;
		}
	}
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
	uint tileRow = gid.x;
	uint tileCol = gid.y;
	uint tid = lid.x;

	uint rowBase = tileRow * BM;
	uint colBase = tileCol * BN;

	// Thread position in the 16x16 grid
	uint threadRow = tid / THREADS_COL;
	uint threadCol = tid % THREADS_COL;

	// Per-thread staging slot: this thread always loads row r, K-cols c..c+3 of each
	// tile (invariant across the K-loop). BM*BK == BN*BK == 256*4, one float4/thread.
	uint r = (tid * 4u) / BK;
	uint c = (tid * 4u) % BK;
	uint shOff = r * BK_PAD + c;
	uint aGRow = rowBase + r;
	uint bGRow = colBase + r;

	// Each thread owns a 4x4 accumulator
	float acc[TM * TN];
	for (uint i = 0; i < TM * TN; i++) acc[i] = 0.0;

	uint aRow = threadRow * TM;
	uint bRow = threadCol * TN;

	// Prefetch tile 0 (K-columns c..c+3) into registers.
	float aReg[4];
	float bReg[4];
	LoadSlice(aGRow, c, pc.a_idx, pc.M, aReg);
	LoadSlice(bGRow, c, pc.b_idx, pc.N, bReg);

	for (uint kBase = 0; kBase < pc.K; kBase += BK) {
		// Publish the prefetched tile to SMEM.
		[[unroll]] for (uint e = 0; e < 4u; ++e) {
			smA[shOff + e] = aReg[e];
			smB[shOff + e] = bReg[e];
		}
		GroupMemoryBarrierWithGroupSync();

		// Issue the NEXT tile's global loads now (into registers) so they fly in the
		// background while the FMAs below run — this is the latency-hiding step.
		uint nextK = kBase + BK;
		if (nextK < pc.K) {
			LoadSlice(aGRow, nextK + c, pc.a_idx, pc.M, aReg);
			LoadSlice(bGRow, nextK + c, pc.b_idx, pc.N, bReg);
		}

		// Compute: each thread does TM x TN dot products over BK from SMEM.
		// Fully unrolled so the compiler can software-pipeline the FMAs — fill
		// the register-tile FMA chain to hide SMEM-read + FMA latency. This is
		// the occupancy/latency-bound hot path on Intel Xe.
		[[unroll]] for (uint k = 0; k < BK; k++) {
			float aVals[TM];
			float bVals[TN];
			[[unroll]] for (uint i = 0; i < TM; i++) aVals[i] = smA[(aRow + i) * BK_PAD + k];
			[[unroll]] for (uint j = 0; j < TN; j++) bVals[j] = smB[(bRow + j) * BK_PAD + k];
			[[unroll]] for (uint i = 0; i < TM; i++) {
				[[unroll]] for (uint j = 0; j < TN; j++) {
					acc[i * TN + j] += aVals[i] * bVals[j];
				}
			}
		}

		GroupMemoryBarrierWithGroupSync();
	}

	// Write back the 4x4 sub-tile
	for (uint i = 0; i < TM; i++) {
		for (uint j = 0; j < TN; j++) {
			uint globalR = rowBase + threadRow * TM + i;
			uint globalC = colBase + threadCol * TN + j;
			if (globalR < pc.M && globalC < pc.N) {
				OaStore(heap[pc.out_idx], globalR * pc.N + globalC, acc[i * TN + j]);
			}
		}
	}
}
