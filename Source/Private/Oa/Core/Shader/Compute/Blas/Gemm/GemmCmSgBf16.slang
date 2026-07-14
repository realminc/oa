// GemmCmSgBf16 — C[i,j] = sum_k A[i,k] * B[j,k]  (B stored transposed: C = A @ B^T)
// The tuned KHR cooperative-matrix GEMM that replaces the GemmCoopMat* zoo.
// Reads FP32 masters → stages BF16 in SMEM → BF16 tensor-core compute → FP32
// accumulator → DTYPE-aware output. No pack, no mirror. Pure VK_KHR_cooperative_
// matrix (16x16x16 subgroup fragments) → runs on NVIDIA and AMD RDNA3.5/Strix.
//
// Recipe adapted from the public vk_cooperative_matrix_perf sample (shaders/shmem.comp, MIT).
// Techniques:
//   [1] 128-bit vectorized global loads (float4 = 4 fp32 / transaction)
//   [2] software-pipelined double-buffer prefetch — overlap next-tile global
//       fetch with current-tile math (the big structural win)
//   [3] bank-pad skew on SMEM rows
//   [4] register tiling — 128x128 output / workgroup, 16 subgroups (4x4),
//       each owning C_ROWS x C_COLS = 2x2 = 4 accumulators (16x16 each).
//       Tuned 2026-07-04: WG_H 2→4 (8→16 subgroups) halves accumulators/thread →
//       ~2× resident warps → Tensor Active 46%↑, +4-5% @2048³ (nsys-confirmed
//       occupancy-bound; see OaGemmRewritePlan.md). Override via -D OA_WG_H=.
//   [5] TILE_K=16 (bf16 ref winner on 5090; override via -D OA_BK=...)
//
// Dispatch grid = ceil(M/128) x ceil(N/128).
//
// Build-time variants via OA_ACT:
//   0 = raw GEMM (direct-to-global store, 3 buffers)
//   1 = + Bias (4 buffers: a, b, bias, out)
//   2 = + Bias + ReLU
//   3 = + Bias + GELU
//   4 = SiLU dual output (pre + act, 4 buffers: a, b, pre, act)
//
// Fused variants use smOut staging so each thread can apply the epilogue
// element-wise with a bounds check. The raw variant uses a direct CoopMat
// store to avoid the extra SMEM/barrier.

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

static const uint CM_M    = 16;
static const uint CM_N    = 16;
static const uint CM_K    = 16;

// Tile geometry — 128x128 output, 64-deep K tile
static const uint WG_W    = 4;    // subgroups in N dimension
#ifndef OA_WG_H
#define OA_WG_H 4
#endif
static const uint WG_H    = OA_WG_H;    // subgroups in M dimension (occupancy lever: 4 halves accumulators/thread vs 2)
static const uint NSUB    = WG_W * WG_H;  // 16 subgroups (4x4)
static const uint BM      = 128;  // output tile M
static const uint BN      = 128;  // output tile N
#ifndef OA_BK
#define OA_BK 64
#endif
static const uint BK      = OA_BK;   // K tile depth (ref uses 16 for bf16; 16/32/64 are a wash on our double-buffered kernel — see OaMatmulOptimization.md §−1)
// Perf experiment: drop per-load bounds checks (assume M,N,K tile-aligned). The
// reference (shmem.comp) has zero bounds checking — it dispatches exact tile
// multiples. Every branch here inflates register pressure and is paid once per
// K-iteration, so it hurts most at small BK. Router must guarantee alignment.
#ifndef OA_ASSUME_ALIGNED
#define OA_ASSUME_ALIGNED 0
#endif
static const uint BK_PAD  = BK + 8;  // skew to break SMEM bank conflicts
static const uint THREADS = NSUB * 32;  // 512 (16 subgroups)
static const uint VEC     = 4;    // float4 = 4 fp32 per global load

// Register tiling: each subgroup owns C_ROWS x C_COLS accumulators
static const uint C_ROWS  = BM / (WG_H * CM_M);  // 128/4/16 = 2 (was 4 at WG_H=2)
static const uint C_COLS  = BN / (WG_W * CM_N);  // 128/4/16 = 2

// Elements per thread for staging: each thread loads VEC fp32 elements
static const uint A_ROWS_PER_THREAD = BM / (THREADS / (BK / VEC));  // rows each thread covers for A
static const uint B_ROWS_PER_THREAD = BN / (THREADS / (BK / VEC));  // rows each thread covers for B

// Native-bf16 staging geometry (DTYPE=1). A 128-bit uint4 load = 8 bf16 / thread —
// KEEP the load 128-bit: a 64-bit uint2 load (4 bf16/thread) was byte-correct but
// regressed 14× on this GPU (halving per-transaction width kills memory throughput;
// see OaMatmulOptimization.md §−1). Because a 128-bit load covers 8 K-elems for bf16
// vs 4 for fp32, the bf16 tiling differs: half the threads per K-row, twice the rows
// per iteration, half the iterations.
static const uint VEC_B     = 8;                 // uint4 = 8 bf16 per 128-bit load
static const uint TPR_B     = BK / VEC_B;        // 8 threads per K-row
static const uint RPI_B     = THREADS / TPR_B;   // 32 rows per staging iteration
static const uint A_ITERS_B = BM / RPI_B;        // 4
static const uint B_ITERS_B = BN / RPI_B;        // 4
static const uint ROW4      = BK_PAD / VEC_B;     // 72/8 = 9 uint4 per padded row

groupshared uint16_t smA[BM * BK_PAD];  // [row][k_col] stride BK_PAD — fp32 path
groupshared uint16_t smB[BN * BK_PAD];  // [n_row][k_col] stride BK_PAD — fp32 path
// Native-bf16 path stores/loads SMEM as uint4 (128-bit) — one wide store per thread,
// matching the reference sample (shmem.comp: `shared uvec4 Ash[]`, coopMatLoad with
// offset/stride ÷ELEMENTS_PER_VEC4). Only one of {smA,smA4} is referenced per DTYPE
// pipeline (spec-constant dead-branch elim), so SMEM footprint is unchanged (36 KB).
groupshared uint4 smA4[BM * ROW4];  // [row]   stride ROW4 uint4 — bf16 path
groupshared uint4 smB4[BN * ROW4];  // [n_row] stride ROW4 uint4 — bf16 path
#if OA_ACT != 0
// Per-subgroup fragment scratch: each of NSUB subgroups gets CM_M*CM_N floats
// (16 * 256 = 4096 floats = 16 KB at WG_H=4). Total SMEM: 36 KB + 16 KB = 52 KB —
// fits Blackwell's opt-in shared-mem ceiling (well above the generic 48 KB). The
// previous smOut[BM*BN] = 64 KB pushed total to 100 KB, exceeding even that and
// causing pipeline creation to defer/hang on first dispatch.
groupshared float    smFrag[NSUB * CM_M * CM_N];
#endif

typealias MatA = CoopMat<BFloat16, MemoryScope.Subgroup, CM_M, CM_K, CoopMatMatrixUse.MatrixA>;
typealias MatB = CoopMat<BFloat16, MemoryScope.Subgroup, CM_K, CM_N, CoopMatMatrixUse.MatrixB>;
typealias MatC = CoopMat<float,    MemoryScope.Subgroup, CM_M, CM_N, CoopMatMatrixUse.MatrixAccumulator>;

uint16_t Fp32ToBf16Bits(float v) {
	return uint16_t(asuint(v) >> 16);
}

// 128-bit (8 bf16) staging load for the native-bf16 path. With OA_ASSUME_ALIGNED
// the bounds check + scalar tail vanish → straight-line uvec4 load like the ref.
// MUST be force-inlined: a real call makes heap[bufIdx] a non-uniform descriptor
// index → the driver falls to a per-thread slow path (measured 9× slower).
[ForceInline]
uint4 LoadBf16Vec(uint bufIdx, uint gR, uint gC, uint rows, uint K) {
#if OA_ASSUME_ALIGNED
	return heap[bufIdx].Load<uint4>((gR * K + gC) * 2);
#else
	if (gR < rows && gC + VEC_B <= K) {
		return heap[bufIdx].Load<uint4>((gR * K + gC) * 2);
	}
	uint4 v = uint4(0u);
    [[unroll]] for (uint e = 0; e < VEC_B; ++e) {
			if (gR < rows && gC + e < K) {
				v[e >> 1] |= uint(Fp32ToBf16Bits(OaLoad(heap[bufIdx], gR * K + gC + e))) << ((e & 1u) * 16u);
			}
	}
	return v;
#endif
}

[shader("compute")]
[numthreads(THREADS, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
	uint tid = lid.x;
	uint sgid = tid / 32;  // subgroup ID within workgroup (0..15)
	uint warp_row = sgid / WG_W;  // 0..3
	uint warp_col = sgid % WG_W;  // 0..3

	uint row_base = gid.x * BM;
	uint col_base = gid.y * BN;

	if (row_base >= pc.M || col_base >= pc.N) return;

	// Register-tiled accumulators: C_ROWS x C_COLS per subgroup
	MatC acc[C_ROWS][C_COLS];
	[[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
		[[unroll]] for (uint j = 0; j < C_COLS; ++j) {
			acc[i][j].fill(0.0f);
		}
	}

	if (OaIsBf16()) {
		// ══ Native-bf16 path: 128-bit uint4 loads (8 bf16 / thread), no fp32→bf16 ══
		//    truncate, HALF the global bytes of the fp32 path. DTYPE is a spec constant
		//    so only this branch compiles into the =1 pipeline (no runtime cost).
		uint4 tempA[A_ITERS_B];
		uint4 tempB[B_ITERS_B];
		{
			uint kk = 0;
			[[unroll]] for (uint i = 0; i < A_ITERS_B; ++i) {
				uint r = i * RPI_B + tid / TPR_B;
				uint c = (tid % TPR_B) * VEC_B;
				tempA[i] = LoadBf16Vec(pc.a_idx, row_base + r, kk + c, pc.M, pc.K);
			}
			[[unroll]] for (uint i = 0; i < B_ITERS_B; ++i) {
				uint r = i * RPI_B + tid / TPR_B;
				uint c = (tid % TPR_B) * VEC_B;
				tempB[i] = LoadBf16Vec(pc.b_idx, col_base + r, kk + c, pc.N, pc.K);
			}
		}

		for (uint kk = 0; kk < pc.K; kk += BK) {
			bool last = (kk + BK >= pc.K);

			// (a) Store temp → SMEM: ONE 128-bit uint4 store per thread (bits already bf16).
			[[unroll]] for (uint i = 0; i < A_ITERS_B; ++i) {
				uint r = i * RPI_B + tid / TPR_B;
				uint c = (tid % TPR_B) * VEC_B;
				smA4[r * ROW4 + c / VEC_B] = tempA[i];
			}
			[[unroll]] for (uint i = 0; i < B_ITERS_B; ++i) {
				uint r = i * RPI_B + tid / TPR_B;
				uint c = (tid % TPR_B) * VEC_B;
				smB4[r * ROW4 + c / VEC_B] = tempB[i];
			}

			GroupMemoryBarrierWithGroupSync();

			// (c) Prefetch NEXT tile (overlaps math).
			if (!last) {
				uint nextKk = kk + BK;
				[[unroll]] for (uint i = 0; i < A_ITERS_B; ++i) {
					uint r = i * RPI_B + tid / TPR_B;
					uint c = (tid % TPR_B) * VEC_B;
					tempA[i] = LoadBf16Vec(pc.a_idx, row_base + r, nextKk + c, pc.M, pc.K);
				}
				[[unroll]] for (uint i = 0; i < B_ITERS_B; ++i) {
					uint r = i * RPI_B + tid / TPR_B;
					uint c = (tid % TPR_B) * VEC_B;
					tempB[i] = LoadBf16Vec(pc.b_idx, col_base + r, nextKk + c, pc.N, pc.K);
				}
			}

			// (d) Math loop from uint4 SMEM — offset/stride ÷VEC_B (uvec4 = 8 bf16), the
			//     same reinterpret the reference does (coopMatLoad ... /ELEMENTS_PER_VEC4).
			[[unroll]] for (uint ku = 0; ku < BK; ku += CM_K) {
				MatA matA[C_ROWS];
				[[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
					uint si = CM_M * (C_ROWS * warp_row + i);
					matA[i] = MatA.Load<CoopMatMatrixLayout.RowMajor>(smA4, (si * BK_PAD + ku) / VEC_B, ROW4);
				}
				[[unroll]] for (uint j = 0; j < C_COLS; ++j) {
					uint sj = CM_N * (C_COLS * warp_col + j);
					MatB b = MatB.Load<CoopMatMatrixLayout.ColumnMajor>(smB4, (sj * BK_PAD + ku) / VEC_B, ROW4);
					[[unroll]] for (uint i = 0; i < C_ROWS; ++i)
						acc[i][j] = coopMatMulAdd<float, false>(matA[i], b, acc[i][j]);
				}
			}

			GroupMemoryBarrierWithGroupSync();
		}
	} else {
		// ── Prefetch tile-0 into registers (double-buffer prologue) ────────────
		float4 tempA[A_ROWS_PER_THREAD];
		float4 tempB[B_ROWS_PER_THREAD];

		{
			uint kk = 0;
			// Prefetch A
			[[unroll]] for (uint i = 0; i < A_ROWS_PER_THREAD; ++i) {
				uint r = i * (THREADS / (BK / VEC)) + tid / (BK / VEC);
				uint c = (tid % (BK / VEC)) * VEC;
				uint gR = row_base + r;
				uint gC = kk + c;
				if (gR < pc.M && gC + VEC <= pc.K) {
					tempA[i] = heap[pc.a_idx].Load<float4>((gR * pc.K + gC) * 4);
				} else {
					tempA[i] = float4(0.0f);
					[[unroll]] for (uint e = 0; e < VEC; ++e) {
						if (gR < pc.M && gC + e < pc.K) {
							tempA[i][e] = OaLoad(heap[pc.a_idx], gR * pc.K + gC + e);
						}
					}
				}
			}
			// Prefetch B
			[[unroll]] for (uint i = 0; i < B_ROWS_PER_THREAD; ++i) {
				uint r = i * (THREADS / (BK / VEC)) + tid / (BK / VEC);
				uint c = (tid % (BK / VEC)) * VEC;
				uint gR = col_base + r;
				uint gC = kk + c;
				if (gR < pc.N && gC + VEC <= pc.K) {
					tempB[i] = heap[pc.b_idx].Load<float4>((gR * pc.K + gC) * 4);
				} else {
					tempB[i] = float4(0.0f);
					[[unroll]] for (uint e = 0; e < VEC; ++e) {
						if (gR < pc.N && gC + e < pc.K) {
							tempB[i][e] = OaLoad(heap[pc.b_idx], gR * pc.K + gC + e);
						}
					}
				}
			}
		}

		// ── Main K-loop with double-buffering ──────────────────────────────────
		for (uint kk = 0; kk < pc.K; kk += BK) {
			bool last = (kk + BK >= pc.K);

			// (a) Store temp_* → SMEM (fp32→bf16 conversion)
			[[unroll]] for (uint i = 0; i < A_ROWS_PER_THREAD; ++i) {
				uint r = i * (THREADS / (BK / VEC)) + tid / (BK / VEC);
				uint c = (tid % (BK / VEC)) * VEC;
				uint shOff = r * BK_PAD + c;
				smA[shOff + 0] = Fp32ToBf16Bits(tempA[i].x);
				smA[shOff + 1] = Fp32ToBf16Bits(tempA[i].y);
				smA[shOff + 2] = Fp32ToBf16Bits(tempA[i].z);
				smA[shOff + 3] = Fp32ToBf16Bits(tempA[i].w);
			}
			[[unroll]] for (uint i = 0; i < B_ROWS_PER_THREAD; ++i) {
				uint r = i * (THREADS / (BK / VEC)) + tid / (BK / VEC);
				uint c = (tid % (BK / VEC)) * VEC;
				uint shOff = r * BK_PAD + c;
				smB[shOff + 0] = Fp32ToBf16Bits(tempB[i].x);
				smB[shOff + 1] = Fp32ToBf16Bits(tempB[i].y);
				smB[shOff + 2] = Fp32ToBf16Bits(tempB[i].z);
				smB[shOff + 3] = Fp32ToBf16Bits(tempB[i].w);
			}

			// (b) Barrier — wait for all threads to finish writing SMEM
			GroupMemoryBarrierWithGroupSync();

			// (c) Prefetch NEXT tile into temp_* (overlaps with math below)
			if (!last) {
				uint nextKk = kk + BK;
				[[unroll]] for (uint i = 0; i < A_ROWS_PER_THREAD; ++i) {
					uint r = i * (THREADS / (BK / VEC)) + tid / (BK / VEC);
					uint c = (tid % (BK / VEC)) * VEC;
					uint gR = row_base + r;
					uint gC = nextKk + c;
					if (gR < pc.M && gC + VEC <= pc.K) {
						tempA[i] = heap[pc.a_idx].Load<float4>((gR * pc.K + gC) * 4);
					} else {
						tempA[i] = float4(0.0f);
						[[unroll]] for (uint e = 0; e < VEC; ++e) {
							if (gR < pc.M && gC + e < pc.K) {
								tempA[i][e] = OaLoad(heap[pc.a_idx], gR * pc.K + gC + e);
							}
						}
					}
				}
				[[unroll]] for (uint i = 0; i < B_ROWS_PER_THREAD; ++i) {
					uint r = i * (THREADS / (BK / VEC)) + tid / (BK / VEC);
					uint c = (tid % (BK / VEC)) * VEC;
					uint gR = col_base + r;
					uint gC = nextKk + c;
					if (gR < pc.N && gC + VEC <= pc.K) {
						tempB[i] = heap[pc.b_idx].Load<float4>((gR * pc.K + gC) * 4);
					} else {
						tempB[i] = float4(0.0f);
						[[unroll]] for (uint e = 0; e < VEC; ++e) {
							if (gR < pc.N && gC + e < pc.K) {
								tempB[i][e] = OaLoad(heap[pc.b_idx], gR * pc.K + gC + e);
							}
						}
					}
				}
			}

			// (d) Math loop — compute from SMEM
			[[unroll]] for (uint ku = 0; ku < BK; ku += CM_K) {
				MatA matA[C_ROWS];
				[[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
					uint si = CM_M * (C_ROWS * warp_row + i);
					matA[i] = MatA.Load<CoopMatMatrixLayout.RowMajor>(smA, si * BK_PAD + ku, BK_PAD);
				}
				[[unroll]] for (uint j = 0; j < C_COLS; ++j) {
					uint sj = CM_N * (C_COLS * warp_col + j);
					MatB b = MatB.Load<CoopMatMatrixLayout.ColumnMajor>(smB, sj * BK_PAD + ku, BK_PAD);
					[[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
						acc[i][j] = coopMatMulAdd<float, false>(matA[i], b, acc[i][j]);
					}
				}
			}

			// (e) Barrier — wait for all threads to finish reading SMEM before
			//     next iteration writes new data
			GroupMemoryBarrierWithGroupSync();
		}
	}  // end DTYPE branch (bf16 native-load vs fp32-master-truncate staging)

	// ── Output store ───────────────────────────────────────────────────────
#if OA_ACT == 0
	// Raw GEMM: direct-to-global store (M%16==0 && N%16==0 contract)
	[[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
		[[unroll]] for (uint j = 0; j < C_COLS; ++j) {
			uint gR = row_base + CM_M * (C_ROWS * warp_row + i);
			uint gC = col_base + CM_N * (C_COLS * warp_col + j);
			if (gR < pc.M && gC < pc.N) {
				acc[i][j].Store<CoopMatMatrixLayout.RowMajor>(
					heap[pc.out_idx], gR * pc.N + gC, pc.N);
			}
		}
	}
#else
	// Fused variants: process one (i,j) fragment at a time. Each subgroup stores
	// its 16x16 accumulator to its own region in smFrag (8KB total), then the 32
	// threads in the subgroup apply the epilogue element-wise (256/32=8 per thread).
	// This avoids the 64KB smOut[BM*BN] that exceeded the 48KB SMEM limit.
	const uint sgBase = sgid * CM_M * CM_N;
	const uint lane = tid % 32;
	[[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
		[[unroll]] for (uint j = 0; j < C_COLS; ++j) {
			acc[i][j].Store<CoopMatMatrixLayout.RowMajor>(smFrag, sgBase, CM_N);
			GroupMemoryBarrierWithGroupSync();

			for (uint t = lane; t < CM_M * CM_N; t += 32) {
				uint lr = t / CM_N;
				uint lc = t % CM_N;
				uint gR = row_base + CM_M * (C_ROWS * warp_row + i) + lr;
				uint gC = col_base + CM_N * (C_COLS * warp_col + j) + lc;
				if (gR < pc.M && gC < pc.N) {
					uint out_idx = gR * pc.N + gC;
					float v = smFrag[sgBase + t];
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
			}
			GroupMemoryBarrierWithGroupSync();
		}
	}
#endif
}
