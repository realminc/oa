// ScatterAddRows: generic atomic indexed-row accumulation.
// MoE token-pack backward uses deterministic MoeGatherBwd instead.
// Source[R,D], UInt32 Indices[R] -> Out[T,D], where Out starts at zero.
//
// One invocation handles one routed scalar, so work is O(R*D). This is the
// correct sparse-MoE adjoint; the deterministic embedding GatherBwd instead
// scans every index for every vocabulary row and is O(T*R*D).

import Storage;

struct PushConstants {
	uint source_idx;
	uint indices_idx;
	uint out_idx;
	uint R;
	uint D;
	uint T;
	uint Dtype;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

// This kernel is also used by explicit BF16 matrices while the engine-wide
// pipeline precision is FP32. Therefore storage is selected from the actual
// matrix dtype carried by the dispatch, not Storage.DTYPE.
[ForceInline]
float LoadTyped(RWByteAddressBuffer buf, uint idx, bool bf16) {
	if (!bf16) return asfloat(buf.Load(idx * 4u));
	uint byteOffset = idx * 2u;
	uint word = buf.Load(byteOffset & ~3u);
	uint shift = (byteOffset & 2u) * 8u;
	return asfloat(((word >> shift) & 0xFFFFu) << 16u);
}

// Storage.OaAtomicAdd deliberately targets FP32 accumulators regardless of
// storage dtype. Sparse routing gradients preserve the source dtype, so BF16
// needs a packed-half CAS rather than treating its storage as FP32.
[ForceInline]
void AtomicAddTyped(RWByteAddressBuffer buf, uint idx, float value, bool bf16) {
	if (!bf16) {
		OaAtomicAdd(buf, idx, value);
		return;
	}
	uint byteOffset = idx * 2u;
	uint wordOffset = byteOffset & ~3u;
	uint shift = (byteOffset & 2u) * 8u;
	uint mask = 0xFFFFu << shift;
	uint oldWord = buf.Load(wordOffset);
	uint assumed;
	do {
		assumed = oldWord;
		uint oldBf16 = (assumed >> shift) & 0xFFFFu;
		float oldValue = asfloat(oldBf16 << 16u);
		uint bits = asuint(oldValue + value);
		uint rounded = bits + 0x7FFFu + ((bits >> 16u) & 1u);
		uint newBf16 = (rounded >> 16u) & 0xFFFFu;
		uint newWord = (assumed & ~mask) | (newBf16 << shift);
		buf.InterlockedCompareExchange(wordOffset, assumed, newWord, oldWord);
	} while (assumed != oldWord);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint i = tid.x;
	if (i >= pc.R * pc.D) return;
	uint route = i / pc.D;
	uint d = i % pc.D;
	uint token = heap[pc.indices_idx].Load(route * 4u);
	if (token >= pc.T) return;
	bool bf16 = pc.Dtype == 1u;
	AtomicAddTyped(heap[pc.out_idx], token * pc.D + d,
		LoadTyped(heap[pc.source_idx], i, bf16), bf16);
}
