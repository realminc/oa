// GemmCoopVec — GEMV using VK_NV_cooperative_vector for decode inference (M=1 or small batch).
// out[i] = sum_k A[i,k] * x[k]  — row-major A, vector x, output y.
// Each subgroup handles one or more output rows via CoopVec matrix-vector multiply.
// Significantly outperforms scalar GEMV at M=1 by using tensor units (Blackwell+).
// Dispatch: (ceil(N / ROWS_PER_WG), 1, 1)

import Storage;
using namespace linalg;

static const uint ROWS_PER_WG = 16;
static const uint VEC_LEN     = 16;
static const uint THREADS      = 32;

struct PushConstants {
	uint a_idx;
	uint x_idx;
	uint out_idx;
	uint N;
	uint K;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

typealias CV = CoopVec<float, VEC_LEN>;

[shader("compute")]
[numthreads(THREADS, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
	uint tid     = lid.x;
	uint row_base = gid.x * ROWS_PER_WG;
	if (row_base >= pc.N) return;

	// Load x vector segment into CV
	for (uint r = 0; r < ROWS_PER_WG; r++) {
		uint row = row_base + r;
		if (row >= pc.N) break;

		float acc = 0.0f;
		for (uint k = tid; k < pc.K; k += THREADS) {
			float a = OaLoad(heap[pc.a_idx], row * pc.K + k);
			float x = OaLoad(heap[pc.x_idx], k);
			acc += a * x;
		}

		// Subgroup reduction
		for (uint offset = THREADS / 2; offset > 0; offset >>= 1) {
			acc += WaveReadLaneAt(acc, WaveGetLaneIndex() ^ offset);
		}

		if (tid == 0 && row < pc.N) {
			OaStore(heap[pc.out_idx], row, acc);
		}
	}
}
