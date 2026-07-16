// Deterministic adjoint for expert-major MoE token packing.
//
// Inverse maps token-major route slot t*K+k to its expert-major packed row.
// One invocation owns one output scalar and sums routes in stable k order, so
// replay is bit-stable and no atomic accumulation or zero-fill pass is needed.

struct PushConstants {
	uint source_idx;
	uint inverse_idx;
	uint out_idx;
	uint R;
	uint D;
	uint T;
	uint K;
	uint Dtype;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[ForceInline]
float LoadTyped(RWByteAddressBuffer buf, uint idx, bool bf16) {
	if (!bf16) return asfloat(buf.Load(idx * 4u));
	uint byteOffset = idx * 2u;
	uint word = buf.Load(byteOffset & ~3u);
	uint shift = (byteOffset & 2u) * 8u;
	return asfloat(((word >> shift) & 0xFFFFu) << 16u);
}

[ForceInline]
void StoreTyped(RWByteAddressBuffer buf, uint idx, float value, bool bf16) {
	if (!bf16) {
		buf.Store(idx * 4u, asuint(value));
		return;
	}
	uint bits = asuint(value);
	uint rounded = bits + 0x7FFFu + ((bits >> 16u) & 1u);
	uint byteOffset = idx * 2u;
	uint shift = (byteOffset & 2u) * 8u;
	uint wordOffset = byteOffset & ~3u;
	uint mask = 0xFFFFu << shift;
	uint oldWord = buf.Load(wordOffset);
	uint assumed;
	do {
		assumed = oldWord;
		uint newWord = (assumed & ~mask) |
			(((rounded >> 16u) & 0xFFFFu) << shift);
		buf.InterlockedCompareExchange(wordOffset, assumed, newWord, oldWord);
	} while (assumed != oldWord);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint i = tid.x;
	if (i >= pc.T * pc.D) return;
	uint token = i / pc.D;
	uint d = i % pc.D;
	bool bf16 = pc.Dtype == 1u;
	float sum = 0.0f;
	for (uint k = 0; k < pc.K; ++k) {
		uint packed = heap[pc.inverse_idx].Load((token * pc.K + k) * 4u);
		if (packed < pc.R)
			sum += LoadTyped(heap[pc.source_idx], packed * pc.D + d, bf16);
	}
	StoreTyped(heap[pc.out_idx], i, sum, bf16);
}
