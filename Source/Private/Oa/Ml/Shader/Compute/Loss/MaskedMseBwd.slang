// Prediction gradient for fused broadcast-masked MSE.

import Storage;

struct PushConstants {
	uint prediction_idx;
	uint target_idx;
	uint mask_idx;
	uint denominator_idx;
	uint upstream_idx;
	uint gradient_idx;
	uint count;
	uint rank;
	uint prediction_strides[8];
	uint mask_dims[8];
	uint mask_strides[8];
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

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
void main(uint3 tid : SV_DispatchThreadID) {
	uint index = tid.x;
	if (index >= pc.count) return;
	float weight = OaLoad(heap[pc.mask_idx], MaskIndex(index));
	float difference = OaLoad(heap[pc.prediction_idx], index)
		- OaLoad(heap[pc.target_idx], index);
	float denominator = OaLoad(heap[pc.denominator_idx], 0u);
	float upstream = OaLoad(heap[pc.upstream_idx], 0u);
	OaStore(heap[pc.gradient_idx], index,
		upstream * 2.0f * difference * weight / denominator);
}
