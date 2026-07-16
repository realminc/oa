// Correctness-complete BLASLt fallback for arbitrary element strides and
// strided batches. Contiguous product shapes continue through OaTile kernels.
import Storage;

struct PushConstants {
	uint a_idx;
	uint b_idx;
	uint out_idx;
	uint M;
	uint N;
	uint K;
	uint aOffset;
	uint aRowStride;
	uint aColStride;
	uint aBatchStride;
	uint bOffset;
	uint bRowStride;
	uint bColStride;
	uint bBatchStride;
	uint cOffset;
	uint cRowStride;
	uint cColStride;
	uint cBatchStride;
};
[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint linear = tid.x;
	if (linear >= pc.M * pc.N) return;
	uint batch = tid.z;
	uint row = linear / pc.N;
	uint col = linear % pc.N;
	uint aBase = pc.aOffset + batch * pc.aBatchStride + row * pc.aRowStride;
	uint bBase = pc.bOffset + batch * pc.bBatchStride + col * pc.bRowStride;
	float sum = 0.0;
	for (uint k = 0; k < pc.K; ++k) {
		sum += OaLoad(heap[pc.a_idx], aBase + k * pc.aColStride)
			* OaLoad(heap[pc.b_idx], bBase + k * pc.bColStride);
	}
	uint dst = pc.cOffset + batch * pc.cBatchStride
		+ row * pc.cRowStride + col * pc.cColStride;
	OaStore(heap[pc.out_idx], dst, sum);
}
