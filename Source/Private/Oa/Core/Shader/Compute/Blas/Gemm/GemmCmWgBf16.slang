// GemmCmWgBf16 — C[i,j] = sum_k A[i,k] * B[j,k]  (B stored transposed: C = A @ B^T)
// KHR cooperative-matrix GEMM using workgroup-scope 32x32x16 fragments.
// Runs on NVIDIA and any device that advertises 32x32x16 workgroup-scope BF16
// CoopMat. Falls back to GemmCmSgBf16 (subgroup-scope 16x16x16) on other hardware.
//
// Dispatch grid = ceil(M/64) x ceil(N/64), one 64x64 output tile per workgroup.
// Each workgroup loads the full 64x64 K-slice into SMEM once, then computes four
// 32x32 workgroup-scope CoopMat blocks. This amortizes global loads and keeps SMEM
// small enough to allow high occupancy (≈34 KB vs ≈90 KB for a 128x128 tile).
//
// Build-time variants via OA_ACT:
//   0 = raw GEMM (direct-to-global store, 3 buffers)
//   1 = + Bias (4 buffers: a, b, bias, out)
//   2 = + Bias + ReLU
//   3 = + Bias + GELU
//   4 = SiLU dual output (pre + act, 4 buffers: a, b, pre, act)

#ifndef OA_ACT
#define OA_ACT 0
#endif

import Storage;
#if OA_ACT > 0 && OA_ACT < 4
import Epilogue;
#endif
#if OA_ACT == 4
import Activations;
#endif
using namespace linalg;

#if OA_ACT == 0
struct FullPushConstants {
	uint a_idx;
	uint b_idx;
	uint out_idx;
	uint M;
	uint N;
	uint K;
};
#else
struct FullPushConstants {
	uint a_idx;
	uint b_idx;
	uint aux_idx;
	uint out_idx;
	uint M;
	uint N;
	uint K;
};
#endif

[[vk::push_constant]] FullPushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

static const uint CM_M    = 32;    // workgroup-scope CoopMat fragment rows
static const uint CM_N    = 32;    // workgroup-scope CoopMat fragment cols
static const uint CM_K    = 16;    // workgroup-scope CoopMat fragment inner

static const uint BM      = 64;    // workgroup output tile rows
static const uint BN      = 64;    // workgroup output tile cols
static const uint BK      = 64;    // K tile depth
static const uint BK_PAD  = BK + 8; // skew to break SMEM bank conflicts
static const uint THREADS = 256;   // one 32x32 workgroup CoopMat uses 256 threads
static const uint VEC     = 4;     // float4 = 4 fp32 per global load

static const uint BLOCKS_M = BM / CM_M;  // 2 rows of 32x32 blocks
static const uint BLOCKS_N = BN / CM_N;  // 2 cols of 32x32 blocks
static const uint A_LOAD_ITERS = (BM * BK) / (THREADS * VEC);  // 4
static const uint B_LOAD_ITERS = (BN * BK) / (THREADS * VEC);  // 4

// Native-bf16 staging (DTYPE=1): 128-bit uint4 load = 8 bf16 / thread, no fp32→bf16
// truncate, half the global bytes. KEEP the load 128-bit — a 64-bit uint2 load
// regressed 14× on GemmCmSgBf16 (see OaMatmulOptimization.md §−1).
static const uint VEC_B        = 8;                            // uint4 = 8 bf16
static const uint A_LOAD_ITERS_B = (BM * BK) / (THREADS * VEC_B);  // 2
static const uint B_LOAD_ITERS_B = (BN * BK) / (THREADS * VEC_B);  // 2

groupshared uint16_t smA[BM * BK_PAD];
groupshared uint16_t smB[BN * BK_PAD];
groupshared float    smOut[BM * BN];

typealias MatA = CoopMat<BFloat16, MemoryScope.Workgroup, CM_M, CM_K, CoopMatMatrixUse.MatrixA>;
typealias MatB = CoopMat<BFloat16, MemoryScope.Workgroup, CM_K, CM_N, CoopMatMatrixUse.MatrixB>;
typealias MatC = CoopMat<float,    MemoryScope.Workgroup, CM_M, CM_N, CoopMatMatrixUse.MatrixAccumulator>;

uint16_t Fp32ToBf16Bits(float v) {
	return uint16_t(asuint(v) >> 16);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
	uint tid = lid.x;

	uint row_base = gid.x * BM;
	uint col_base = gid.y * BN;

	if (row_base >= pc.M || col_base >= pc.N) return;

	MatC acc[BLOCKS_M][BLOCKS_N];
	[[unroll]] for (uint i = 0; i < BLOCKS_M; ++i)
		[[unroll]] for (uint j = 0; j < BLOCKS_N; ++j)
			acc[i][j].fill(0.0f);

	for (uint kk = 0; kk < pc.K; kk += BK) {
		// Load full A tile for this workgroup: [row_base, row_base+BM) x [kk, kk+BK)
		if (OaIsBf16()) {
			// Native bf16: 128-bit uint4 = 8 bf16 / thread (no truncate, half the bytes).
			[[unroll]] for (uint i = 0; i < A_LOAD_ITERS_B; ++i) {
				uint elemIdx = (i * THREADS + tid) * VEC_B;
				uint r = elemIdx / BK;
				uint c = elemIdx % BK;
				uint gR = row_base + r;
				uint gC = kk + c;
				uint shOff = r * BK_PAD + c;
				if (gR + 1 <= pc.M && gC + VEC_B <= pc.K) {
					uint4 v = heap[pc.a_idx].Load<uint4>((gR * pc.K + gC) * 2);
					smA[shOff + 0] = uint16_t(v.x & 0xFFFFu);  smA[shOff + 1] = uint16_t(v.x >> 16);
					smA[shOff + 2] = uint16_t(v.y & 0xFFFFu);  smA[shOff + 3] = uint16_t(v.y >> 16);
					smA[shOff + 4] = uint16_t(v.z & 0xFFFFu);  smA[shOff + 5] = uint16_t(v.z >> 16);
					smA[shOff + 6] = uint16_t(v.w & 0xFFFFu);  smA[shOff + 7] = uint16_t(v.w >> 16);
				} else {
					[[unroll]] for (uint e = 0; e < VEC_B; ++e)
						smA[shOff + e] = (gR < pc.M && gC + e < pc.K) ? Fp32ToBf16Bits(OaLoad(heap[pc.a_idx], gR * pc.K + gC + e)) : uint16_t(0);
				}
			}
		} else {
		[[unroll]] for (uint i = 0; i < A_LOAD_ITERS; ++i) {
			uint elemIdx = (i * THREADS + tid) * VEC;
			uint r = elemIdx / BK;
			uint c = elemIdx % BK;
			uint gR = row_base + r;
			uint gC = kk + c;
			uint shOff = r * BK_PAD + c;
			if (gR + 1 <= pc.M && gC + VEC <= pc.K) {
				float4 v = heap[pc.a_idx].Load<float4>((gR * pc.K + gC) * 4);
				smA[shOff + 0] = Fp32ToBf16Bits(v.x);
				smA[shOff + 1] = Fp32ToBf16Bits(v.y);
				smA[shOff + 2] = Fp32ToBf16Bits(v.z);
				smA[shOff + 3] = Fp32ToBf16Bits(v.w);
			} else {
				[[unroll]] for (uint e = 0; e < VEC; ++e) {
					if (gR < pc.M && gC + e < pc.K) {
						smA[shOff + e] = Fp32ToBf16Bits(OaLoad(heap[pc.a_idx], gR * pc.K + gC + e));
					} else {
						smA[shOff + e] = 0;
					}
				}
			}
		}
		}
		// Load full B tile for this workgroup: [col_base, col_base+BN) x [kk, kk+BK)
		if (OaIsBf16()) {
			[[unroll]] for (uint i = 0; i < B_LOAD_ITERS_B; ++i) {
				uint elemIdx = (i * THREADS + tid) * VEC_B;
				uint r = elemIdx / BK;
				uint c = elemIdx % BK;
				uint gR = col_base + r;
				uint gC = kk + c;
				uint shOff = r * BK_PAD + c;
				if (gR + 1 <= pc.N && gC + VEC_B <= pc.K) {
					uint4 v = heap[pc.b_idx].Load<uint4>((gR * pc.K + gC) * 2);
					smB[shOff + 0] = uint16_t(v.x & 0xFFFFu);  smB[shOff + 1] = uint16_t(v.x >> 16);
					smB[shOff + 2] = uint16_t(v.y & 0xFFFFu);  smB[shOff + 3] = uint16_t(v.y >> 16);
					smB[shOff + 4] = uint16_t(v.z & 0xFFFFu);  smB[shOff + 5] = uint16_t(v.z >> 16);
					smB[shOff + 6] = uint16_t(v.w & 0xFFFFu);  smB[shOff + 7] = uint16_t(v.w >> 16);
				} else {
					[[unroll]] for (uint e = 0; e < VEC_B; ++e)
						smB[shOff + e] = (gR < pc.N && gC + e < pc.K) ? Fp32ToBf16Bits(OaLoad(heap[pc.b_idx], gR * pc.K + gC + e)) : uint16_t(0);
				}
			}
		} else {
		[[unroll]] for (uint i = 0; i < B_LOAD_ITERS; ++i) {
			uint elemIdx = (i * THREADS + tid) * VEC;
			uint r = elemIdx / BK;
			uint c = elemIdx % BK;
			uint gR = col_base + r;
			uint gC = kk + c;
			uint shOff = r * BK_PAD + c;
			if (gR + 1 <= pc.N && gC + VEC <= pc.K) {
				float4 v = heap[pc.b_idx].Load<float4>((gR * pc.K + gC) * 4);
				smB[shOff + 0] = Fp32ToBf16Bits(v.x);
				smB[shOff + 1] = Fp32ToBf16Bits(v.y);
				smB[shOff + 2] = Fp32ToBf16Bits(v.z);
				smB[shOff + 3] = Fp32ToBf16Bits(v.w);
			} else {
				[[unroll]] for (uint e = 0; e < VEC; ++e) {
					if (gR < pc.N && gC + e < pc.K) {
						smB[shOff + e] = Fp32ToBf16Bits(OaLoad(heap[pc.b_idx], gR * pc.K + gC + e));
					} else {
						smB[shOff + e] = 0;
					}
				}
			}
		}
		}
		GroupMemoryBarrierWithGroupSync();

		[[unroll]] for (uint bi = 0; bi < BLOCKS_M; ++bi) {
			[[unroll]] for (uint bj = 0; bj < BLOCKS_N; ++bj) {
				[[unroll]] for (uint ku = 0; ku < BK; ku += CM_K) {
					MatA a = MatA.Load<CoopMatMatrixLayout.RowMajor>(
						smA, (bi * CM_M) * BK_PAD + ku, BK_PAD);
					MatB b = MatB.Load<CoopMatMatrixLayout.ColumnMajor>(
						smB, (bj * CM_N) * BK_PAD + ku, BK_PAD);
					acc[bi][bj] = coopMatMulAdd<float, false>(a, b, acc[bi][bj]);
				}
			}
		}
		GroupMemoryBarrierWithGroupSync();
	}

	// Store the 2x2 result blocks
	[[unroll]] for (uint bi = 0; bi < BLOCKS_M; ++bi) {
		[[unroll]] for (uint bj = 0; bj < BLOCKS_N; ++bj) {
			uint block_row = row_base + bi * CM_M;
			uint block_col = col_base + bj * CM_N;
			if (block_row >= pc.M || block_col >= pc.N) continue;
			if (block_row + CM_M <= pc.M && block_col + CM_N <= pc.N) {
#if OA_ACT == 0
				acc[bi][bj].Store<CoopMatMatrixLayout.RowMajor>(
					heap[pc.out_idx], block_row * pc.N + block_col, pc.N);
#else
				acc[bi][bj].Store<CoopMatMatrixLayout.RowMajor>(smOut, 0, CM_N);
				GroupMemoryBarrierWithGroupSync();
				for (uint t = tid; t < CM_M * CM_N; t += THREADS) {
					uint r  = t / CM_N;
					uint c  = t % CM_N;
					uint gR = block_row + r;
					uint gC = block_col + c;
					uint out_idx = gR * pc.N + gC;
					float v = smOut[t];
#if OA_ACT == 1
					EpilogueBias(heap[pc.out_idx], heap[pc.aux_idx], out_idx, gC, v);
#elif OA_ACT == 2
					EpilogueBiasRelu(heap[pc.out_idx], heap[pc.aux_idx], out_idx, gC, v);
#elif OA_ACT == 3
					EpilogueBiasGelu(heap[pc.out_idx], heap[pc.aux_idx], out_idx, gC, v);
#elif OA_ACT == 4
					OaStore(heap[pc.aux_idx], out_idx, v);
					OaStore(heap[pc.out_idx], out_idx, Silu(v));
#endif
				}
				GroupMemoryBarrierWithGroupSync();
#endif
			} else {
				// Tail path: store through SMEM to handle partial blocks
				acc[bi][bj].Store<CoopMatMatrixLayout.RowMajor>(smOut, 0, CM_N);
				GroupMemoryBarrierWithGroupSync();
				for (uint t = tid; t < CM_M * CM_N; t += THREADS) {
					uint r  = t / CM_N;
					uint c  = t % CM_N;
					uint gR = block_row + r;
					uint gC = block_col + c;
					if (gR < pc.M && gC < pc.N) {
						uint out_idx = gR * pc.N + gC;
						float v = smOut[t];
#if OA_ACT == 0
						OaStore(heap[pc.out_idx], out_idx, v);
#else
#if OA_ACT == 1
						EpilogueBias(heap[pc.out_idx], heap[pc.aux_idx], out_idx, gC, v);
#elif OA_ACT == 2
						EpilogueBiasRelu(heap[pc.out_idx], heap[pc.aux_idx], out_idx, gC, v);
#elif OA_ACT == 3
						EpilogueBiasGelu(heap[pc.out_idx], heap[pc.aux_idx], out_idx, gC, v);
#elif OA_ACT == 4
						OaStore(heap[pc.aux_idx], out_idx, v);
						OaStore(heap[pc.out_idx], out_idx, Silu(v));
#endif
#endif
					}
				}
				GroupMemoryBarrierWithGroupSync();
			}
		}
	}
}
