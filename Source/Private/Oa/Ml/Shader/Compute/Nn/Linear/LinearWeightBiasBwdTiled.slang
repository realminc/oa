// Cooperative dense-Linear parameter adjoint for large reduction shapes.
// dW[n,k] = sum_m dY[m,n] * X[m,k]
// dB[n]   = sum_m dY[m,n]
//
// The scalar kernel remains preferable for tiny N*K. This variant tiles N/K
// and reuses each staged dY/X reduction slab across a 32x32 output tile.
import Storage;

struct PushConstants {
	uint d_out_idx;
	uint input_idx;
	uint d_weight_idx;
	uint d_bias_idx;
	uint M;
	uint N;
	uint K;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

static const uint BN = 32, BK = 32, BM = 16;
static const uint TN = 4, TK = 4, BM_PAD = BM + 1;
groupshared float smY[BN * BM_PAD];
groupshared float smX[BK * BM_PAD];

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
	uint nBase = gid.x * BN;
	uint kBase = gid.y * BK;
	if (nBase >= pc.N || kBase >= pc.K) return;

	uint tid = lid.x;
	uint threadN = tid / 8;
	uint threadK = tid % 8;
	float acc[TN * TK];
	float biasAcc[TN];
	[[unroll]] for (uint i = 0; i < TN * TK; ++i) acc[i] = 0.0;
	[[unroll]] for (uint i = 0; i < TN; ++i) biasAcc[i] = 0.0;

	for (uint mBase = 0; mBase < pc.M; mBase += BM) {
		[[unroll]] for (uint q = 0; q < 8; ++q) {
			uint idx = tid + q * 64;
			uint outer = idx / BM;
			uint mm = idx % BM;
			uint m = mBase + mm;
			uint n = nBase + outer;
			smY[outer * BM_PAD + mm] = (n < pc.N && m < pc.M)
				? OaLoad(heap[pc.d_out_idx], m * pc.N + n) : 0.0;
			uint k = kBase + outer;
			smX[outer * BM_PAD + mm] = (k < pc.K && m < pc.M)
				? OaLoad(heap[pc.input_idx], m * pc.K + k) : 0.0;
		}
		GroupMemoryBarrierWithGroupSync();

		[[unroll]] for (uint mm = 0; mm < BM; ++mm) {
			float yv[TN], xv[TK];
			[[unroll]] for (uint i = 0; i < TN; ++i)
				yv[i] = smY[(threadN * TN + i) * BM_PAD + mm];
			[[unroll]] for (uint j = 0; j < TK; ++j)
				xv[j] = smX[(threadK * TK + j) * BM_PAD + mm];
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
				OaStore(heap[pc.d_weight_idx], n * pc.K + k, acc[i * TK + j]);
		}
		if (threadK == 0 && kBase == 0) {
			uint n = nBase + threadN * TN + i;
			if (n < pc.N)
				OaStore(heap[pc.d_bias_idx], n, biasAcc[i]);
		}
	}
}
