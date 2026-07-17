// Fused broadcast-masked MSE reduction.
// Output 0 is the scalar loss; output 1 is the broadcast mask denominator
// retained for the matching backward kernel. Dispatch one workgroup.

import Storage;

struct PushConstants {
	uint prediction_idx;
	uint target_idx;
	uint mask_idx;
	uint loss_idx;
	uint denominator_idx;
	uint count;
	uint rank;
	uint prediction_strides[8];
	uint mask_dims[8];
	uint mask_strides[8];
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

groupshared float g_error[256];
groupshared float g_weight[256];

[ForceInline]
uint MaskIndex(uint flat_index) {
	uint remaining = flat_index;
	uint mask_index = 0u;
	[unroll] for (uint dim = 0u; dim < 8u; ++dim) {
		if (dim >= pc.rank) break;
		uint stride = pc.prediction_strides[dim];
		uint coordinate = remaining / stride;
		remaining -= coordinate * stride;
		if (pc.mask_dims[dim] != 1u) {
			mask_index += coordinate * pc.mask_strides[dim];
		}
	}
	return mask_index;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_GroupThreadID) {
	uint lane = tid.x;
	float error_sum = 0.0f;
	float weight_sum = 0.0f;
	for (uint index = lane; index < pc.count; index += 256u) {
		float weight = OaLoad(heap[pc.mask_idx], MaskIndex(index));
		float difference = OaLoad(heap[pc.prediction_idx], index)
			- OaLoad(heap[pc.target_idx], index);
		error_sum += difference * difference * weight;
		weight_sum += weight;
	}
	g_error[lane] = error_sum;
	g_weight[lane] = weight_sum;
	GroupMemoryBarrierWithGroupSync();

	[unroll] for (uint stride = 128u; stride > 0u; stride >>= 1u) {
		if (lane < stride) {
			g_error[lane] += g_error[lane + stride];
			g_weight[lane] += g_weight[lane + stride];
		}
		GroupMemoryBarrierWithGroupSync();
	}

	if (lane == 0u) {
		float denominator = max(g_weight[0], 1.0f);
		OaStore(heap[pc.loss_idx], 0u, g_error[0] / denominator);
		OaStore(heap[pc.denominator_idx], 0u, denominator);
	}
}
