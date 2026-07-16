// LinearDataBwd — tiled data adjoint for a dense Linear.
// dX[m,k] = sum_n dY[m,n] * W[n,k]
//
// This is not expressible by the forward A @ B^T kernel without transposing
// W. Stage dY and W directly instead of paying a transpose or the old scalar
// one-thread-per-output loop.
import Storage;

struct PushConstants {
	uint d_out_idx;
	uint weight_idx;
	uint d_input_idx;
	uint M;
	uint N;
	uint K;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

static const uint BM = 32, BK = 32, BN = 16;
static const uint TM = 4, TK = 4, BN_PAD = BN + 1;
groupshared float smY[BM * BN_PAD];
groupshared float smW[BK * BN_PAD];

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
	uint rowBase = gid.x * BM;
	uint colBase = gid.y * BK;
	if (rowBase >= pc.M || colBase >= pc.K) return;

	uint tid = lid.x;
	uint threadRow = tid / 8;
	uint threadCol = tid % 8;
	float acc[TM * TK];
	[[unroll]] for (uint i = 0; i < TM * TK; ++i) acc[i] = 0.0;

	for (uint nBase = 0; nBase < pc.N; nBase += BN) {
		[[unroll]] for (uint q = 0; q < 8; ++q) {
			uint idx = tid + q * 64;
			uint outLane = idx / BN;
			uint nn = idx % BN;
			uint m = rowBase + outLane;
			uint n = nBase + nn;
			smY[outLane * BN_PAD + nn] = (m < pc.M && n < pc.N)
				? OaLoad(heap[pc.d_out_idx], m * pc.N + n) : 0.0;
			uint k = colBase + outLane;
			smW[outLane * BN_PAD + nn] = (k < pc.K && n < pc.N)
				? OaLoad(heap[pc.weight_idx], n * pc.K + k) : 0.0;
		}
		GroupMemoryBarrierWithGroupSync();

		[[unroll]] for (uint nn = 0; nn < BN; ++nn) {
			float yv[TM], wv[TK];
			[[unroll]] for (uint i = 0; i < TM; ++i)
				yv[i] = smY[(threadRow * TM + i) * BN_PAD + nn];
			[[unroll]] for (uint j = 0; j < TK; ++j)
				wv[j] = smW[(threadCol * TK + j) * BN_PAD + nn];
			[[unroll]] for (uint i = 0; i < TM; ++i)
				[[unroll]] for (uint j = 0; j < TK; ++j)
					acc[i * TK + j] += yv[i] * wv[j];
		}
		GroupMemoryBarrierWithGroupSync();
	}

	[[unroll]] for (uint i = 0; i < TM; ++i) {
		[[unroll]] for (uint j = 0; j < TK; ++j) {
			uint m = rowBase + threadRow * TM + i;
			uint k = colBase + threadCol * TK + j;
			if (m < pc.M && k < pc.K)
				OaStore(heap[pc.d_input_idx], m * pc.K + k, acc[i * TK + j]);
		}
	}
}
