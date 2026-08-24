// Portable FP32 small-M matrix multiplication.
//
// C[row,col] = sum_k A[row,k] * B[col,k], with B stored [N,K]. One
// compile-time reduction group owns one output element. The reduction uses
// groupshared memory rather than subgroup intrinsics, so correctness does not
// depend on a 32-lane hardware subgroup.

import storage;
import epilogue;

#ifndef OA_SMALL_M_OUTPUTS
#define OA_SMALL_M_OUTPUTS 4
#endif
#ifndef OA_SMALL_M_REDUCTION
#define OA_SMALL_M_REDUCTION 32
#endif
#ifndef OA_SMALL_M_WORKGROUP
#define OA_SMALL_M_WORKGROUP 128
#endif
#ifndef OA_SMALL_M_EPILOGUE
#define OA_SMALL_M_EPILOGUE 0
#endif

struct FullPushConstants {
	uint a_idx;
	uint b_idx;
#if OA_SMALL_M_EPILOGUE > 0
	uint bias_idx;
#endif
	uint out_idx;
	uint M;
	uint N;
	uint K;
};

[[vk::push_constant]] FullPushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

static const uint OUTPUTS_PER_GROUP = OA_SMALL_M_OUTPUTS;
static const uint REDUCTION_WIDTH = OA_SMALL_M_REDUCTION;
groupshared float partial[OA_SMALL_M_WORKGROUP];

[shader("compute")]
[numthreads(OA_SMALL_M_WORKGROUP, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
	const uint outputInGroup = lid.x / REDUCTION_WIDTH;
	const uint lane = lid.x % REDUCTION_WIDTH;
	const uint row = gid.x;
	const uint col = gid.y * OUTPUTS_PER_GROUP + outputInGroup;

	float sum = 0.0;
	if (row < pc.M && col < pc.N) {
		for (uint k = lane; k < pc.K; k += REDUCTION_WIDTH) {
			sum += oaLoad(heap[pc.a_idx], row * pc.K + k)
				* oaLoad(heap[pc.b_idx], col * pc.K + k);
		}
	}
	partial[lid.x] = sum;
	GroupMemoryBarrierWithGroupSync();

	for (uint stride = REDUCTION_WIDTH / 2u; stride > 0u; stride >>= 1u) {
		if (lane < stride) {
			partial[lid.x] += partial[lid.x + stride];
		}
		GroupMemoryBarrierWithGroupSync();
	}

	if (lane == 0u && row < pc.M && col < pc.N) {
		const uint outputIndex = row * pc.N + col;
#if OA_SMALL_M_EPILOGUE == 1
		EpilogueBias(heap[pc.out_idx], heap[pc.bias_idx],
			outputIndex, col, partial[lid.x]);
#elif OA_SMALL_M_EPILOGUE == 2
		EpilogueBiasRelu(heap[pc.out_idx], heap[pc.bias_idx],
			outputIndex, col, partial[lid.x]);
#elif OA_SMALL_M_EPILOGUE == 3
		EpilogueBiasGelu(heap[pc.out_idx], heap[pc.bias_idx],
			outputIndex, col, partial[lid.x]);
#elif OA_SMALL_M_EPILOGUE == 4
		EpilogueBiasSilu(heap[pc.out_idx], heap[pc.bias_idx],
			outputIndex, col, partial[lid.x]);
#else
		oaStore(heap[pc.out_idx], outputIndex, partial[lid.x]);
#endif
	}
}
