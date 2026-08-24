// Row-parallel dense-Linear parameter adjoint for narrow feature dimensions.
// One 32x8 workgroup reduces eight independent M stripes for one output row
// and one 32-column input tile, then combines them without global scratch.
//
// dW[n,k] = sum_m dY[m,n] * X[m,k]
// dB[n]   = sum_m dY[m,n]
import storage;

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

static const uint TILE_K = 32;
static const uint REDUCE_Y = 8;
static const uint PADDED_K = TILE_K + 1;

groupshared float smWeight[REDUCE_Y * PADDED_K];
groupshared float smBias[REDUCE_Y];

[shader("compute")]
[numthreads(TILE_K, REDUCE_Y, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
	uint n = gid.x;
	if (n >= pc.N) return;

	uint k = gid.y * TILE_K + lid.x;
	bool validK = k < pc.K;
	float weightSum = 0.0;
	float biasSum = 0.0;
	for (uint m = lid.y; m < pc.M; m += REDUCE_Y) {
		float dy = oaLoad(heap[pc.d_out_idx], m * pc.N + n);
		if (validK) {
			weightSum += dy * oaLoad(heap[pc.input_idx], m * pc.K + k);
		}
		if (lid.x == 0) biasSum += dy;
	}

	smWeight[lid.y * PADDED_K + lid.x] = weightSum;
	if (lid.x == 0) smBias[lid.y] = biasSum;
	GroupMemoryBarrierWithGroupSync();

	if (lid.y == 0) {
		float total = 0.0;
		[[unroll]] for (uint stripe = 0; stripe < REDUCE_Y; ++stripe) {
			total += smWeight[stripe * PADDED_K + lid.x];
		}
		if (validK) oaStore(heap[pc.d_weight_idx], n * pc.K + k, total);

		if (lid.x == 0 && gid.y == 0) {
			float biasTotal = 0.0;
			[[unroll]] for (uint stripe = 0; stripe < REDUCE_Y; ++stripe) {
				biasTotal += smBias[stripe];
			}
			oaStore(heap[pc.d_bias_idx], n, biasTotal);
		}
	}
}
