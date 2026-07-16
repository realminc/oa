// C[i,j] = sum_k A[i,k] * B[j,k]  (B is transposed: C = A @ B^T)
// Generated tiled shared-memory matmul. The established production geometry is
// 64x64x16 with 256 threads and a 4x4 per-thread sub-tile; OaTile may compile
// other validated geometries from the same reviewed arithmetic template.
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
import Epilogue;

#ifndef OA_TILE_GROUP_M
#define OA_TILE_GROUP_M 0
#endif
#ifndef OA_TILE_M
#define OA_TILE_M 64
#endif
#ifndef OA_TILE_N
#define OA_TILE_N 64
#endif
#ifndef OA_TILE_K
#define OA_TILE_K 16
#endif
#ifndef OA_THREAD_TILE_M
#define OA_THREAD_TILE_M 4
#endif
#ifndef OA_THREAD_TILE_N
#define OA_THREAD_TILE_N 4
#endif
#ifndef OA_TILE_WORKGROUP
#define OA_TILE_WORKGROUP 256
#endif
#ifndef OA_TILE_EPILOGUE
#define OA_TILE_EPILOGUE 0
#endif

// Preserve the established 64x64 kernel's exact load/publish instruction shape.
// OaTile may instantiate other legal geometries from this source, but abstraction
// must not make the production control kernel pay for the wider lattice.
#if OA_TILE_M == 64 && OA_TILE_N == 64 && OA_TILE_K == 16 && OA_THREAD_TILE_M == 4 && OA_THREAD_TILE_N == 4 && OA_TILE_WORKGROUP == 256
#define OA_TILE_LEGACY_FAST_PATH 1
#else
#define OA_TILE_LEGACY_FAST_PATH 0
#endif

struct FullPushConstants {
	uint a_idx;
	uint b_idx;
#if OA_TILE_EPILOGUE > 0
	uint bias_idx;
#endif
	uint out_idx;
	uint M;
	uint N;
	uint K;
};

[[vk::push_constant]] FullPushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

static const uint BM = OA_TILE_M;
static const uint BN = OA_TILE_N;
static const uint BK = OA_TILE_K;
static const uint TM = OA_THREAD_TILE_M;
static const uint TN = OA_THREAD_TILE_N;
static const uint WORKGROUP_THREADS = OA_TILE_WORKGROUP;
static const uint GROUP_M = OA_TILE_GROUP_M;

// The generator validates that this 2D logical grid exactly covers the actual
// one-dimensional workgroup and that both shared tiles distribute as float4s.
static const uint THREADS_ROW = BM / TM;
static const uint THREADS_COL = BN / TN;
static const uint LOAD_WIDTH = 4;
static const uint A_LOADS = (BM * BK) / (WORKGROUP_THREADS * LOAD_WIDTH);
static const uint B_LOADS = (BN * BK) / (WORKGROUP_THREADS * LOAD_WIDTH);

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
[numthreads(OA_TILE_WORKGROUP, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
	uint tileRow = gid.x;
	uint tileCol = gid.y;
#if OA_TILE_GROUP_M > 0
	// Group output tiles along M before advancing N. Adjacent workgroups reuse
	// the same B rows while walking a short band of A rows, increasing the
	// chance that both operands remain in the shared LLC/L3. The mapping is a
	// bijection over the existing 2D dispatch grid, so tails and numerics are
	// identical to the row-major control variant.
	const uint tilesM = (pc.M + BM - 1u) / BM;
	const uint tilesN = (pc.N + BN - 1u) / BN;
	const uint linearId = gid.x + gid.y * tilesM;
	const uint groupSpan = GROUP_M * tilesN;
	const uint groupId = linearId / groupSpan;
	const uint firstTileM = groupId * GROUP_M;
	const uint groupSizeM = min(tilesM - firstTileM, GROUP_M);
	const uint inGroup = linearId % groupSpan;
	tileRow = firstTileM + (inGroup % groupSizeM);
	tileCol = inGroup / groupSizeM;
#endif
	uint tid = lid.x;

	uint rowBase = tileRow * BM;
	uint colBase = tileCol * BN;

	// Thread position in the 16x16 grid
	uint threadRow = tid / THREADS_COL;
	uint threadCol = tid % THREADS_COL;

	// Each thread owns a compile-time TM x TN accumulator.
	float acc[TM * TN];
	for (uint i = 0; i < TM * TN; i++) acc[i] = 0.0;

	uint aRow = threadRow * TM;
	uint bRow = threadCol * TN;

	// Keep the established production kernel structurally identical. Wider
	// generated candidates use the generic distributed float4 loader below.
#if OA_TILE_LEGACY_FAST_PATH
	const uint r = (tid * LOAD_WIDTH) / BK;
	const uint c = (tid * LOAD_WIDTH) % BK;
	const uint shOff = r * BK_PAD + c;
	const uint aGRow = rowBase + r;
	const uint bGRow = colBase + r;
	float aReg[LOAD_WIDTH];
	float bReg[LOAD_WIDTH];
	LoadSlice(aGRow, c, pc.a_idx, pc.M, aReg);
	LoadSlice(bGRow, c, pc.b_idx, pc.N, bReg);
#else
	// Non-legacy geometries may need multiple loads per operand. These counts
	// remain compile-time so prefetch can still overlap the FMAs.
	float aReg[A_LOADS * LOAD_WIDTH];
	float bReg[B_LOADS * LOAD_WIDTH];
	[[unroll]] for (uint load = 0; load < A_LOADS; ++load) {
		const uint linear = (tid + load * WORKGROUP_THREADS) * LOAD_WIDTH;
		const uint r = linear / BK;
		const uint c = linear % BK;
		float slice[LOAD_WIDTH];
		LoadSlice(rowBase + r, c, pc.a_idx, pc.M, slice);
		[[unroll]] for (uint e = 0; e < LOAD_WIDTH; ++e) {
			aReg[load * LOAD_WIDTH + e] = slice[e];
		}
	}
	[[unroll]] for (uint load = 0; load < B_LOADS; ++load) {
		const uint linear = (tid + load * WORKGROUP_THREADS) * LOAD_WIDTH;
		const uint r = linear / BK;
		const uint c = linear % BK;
		float slice[LOAD_WIDTH];
		LoadSlice(colBase + r, c, pc.b_idx, pc.N, slice);
		[[unroll]] for (uint e = 0; e < LOAD_WIDTH; ++e) {
			bReg[load * LOAD_WIDTH + e] = slice[e];
		}
	}
#endif

	for (uint kBase = 0; kBase < pc.K; kBase += BK) {
		// Publish the prefetched tile to SMEM.
#if OA_TILE_LEGACY_FAST_PATH
		[[unroll]] for (uint e = 0; e < LOAD_WIDTH; ++e) {
			smA[shOff + e] = aReg[e];
			smB[shOff + e] = bReg[e];
		}
#else
		[[unroll]] for (uint load = 0; load < A_LOADS; ++load) {
			const uint linear = (tid + load * WORKGROUP_THREADS) * LOAD_WIDTH;
			const uint r = linear / BK;
			const uint c = linear % BK;
			[[unroll]] for (uint e = 0; e < LOAD_WIDTH; ++e) {
				smA[r * BK_PAD + c + e] = aReg[load * LOAD_WIDTH + e];
			}
		}
		[[unroll]] for (uint load = 0; load < B_LOADS; ++load) {
			const uint linear = (tid + load * WORKGROUP_THREADS) * LOAD_WIDTH;
			const uint r = linear / BK;
			const uint c = linear % BK;
			[[unroll]] for (uint e = 0; e < LOAD_WIDTH; ++e) {
				smB[r * BK_PAD + c + e] = bReg[load * LOAD_WIDTH + e];
			}
		}
#endif
		GroupMemoryBarrierWithGroupSync();

		// Issue the NEXT tile's global loads now (into registers) so they fly in the
		// background while the FMAs below run — this is the latency-hiding step.
		uint nextK = kBase + BK;
		if (nextK < pc.K) {
#if OA_TILE_LEGACY_FAST_PATH
			LoadSlice(aGRow, nextK + c, pc.a_idx, pc.M, aReg);
			LoadSlice(bGRow, nextK + c, pc.b_idx, pc.N, bReg);
#else
			[[unroll]] for (uint load = 0; load < A_LOADS; ++load) {
				const uint linear = (tid + load * WORKGROUP_THREADS) * LOAD_WIDTH;
				const uint r = linear / BK;
				const uint c = linear % BK;
				float slice[LOAD_WIDTH];
				LoadSlice(rowBase + r, nextK + c, pc.a_idx, pc.M, slice);
				[[unroll]] for (uint e = 0; e < LOAD_WIDTH; ++e) {
					aReg[load * LOAD_WIDTH + e] = slice[e];
				}
			}
			[[unroll]] for (uint load = 0; load < B_LOADS; ++load) {
				const uint linear = (tid + load * WORKGROUP_THREADS) * LOAD_WIDTH;
				const uint r = linear / BK;
				const uint c = linear % BK;
				float slice[LOAD_WIDTH];
				LoadSlice(colBase + r, nextK + c, pc.b_idx, pc.N, slice);
				[[unroll]] for (uint e = 0; e < LOAD_WIDTH; ++e) {
					bReg[load * LOAD_WIDTH + e] = slice[e];
				}
			}
#endif
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

	// Write back the compile-time thread sub-tile.
	for (uint i = 0; i < TM; i++) {
		for (uint j = 0; j < TN; j++) {
			uint globalR = rowBase + threadRow * TM + i;
			uint globalC = colBase + threadCol * TN + j;
			if (globalR < pc.M && globalC < pc.N) {
#if OA_TILE_EPILOGUE == 1
				EpilogueBias(heap[pc.out_idx], heap[pc.bias_idx],
					globalR * pc.N + globalC, globalC, acc[i * TN + j]);
#elif OA_TILE_EPILOGUE == 2
				EpilogueBiasRelu(heap[pc.out_idx], heap[pc.bias_idx],
					globalR * pc.N + globalC, globalC, acc[i * TN + j]);
#elif OA_TILE_EPILOGUE == 3
				EpilogueBiasGelu(heap[pc.out_idx], heap[pc.bias_idx],
					globalR * pc.N + globalC, globalC, acc[i * TN + j]);
#else
				OaStore(heap[pc.out_idx], globalR * pc.N + globalC, acc[i * TN + j]);
#endif
			}
		}
	}
}
