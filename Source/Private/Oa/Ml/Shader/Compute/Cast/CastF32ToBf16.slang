// CastF32ToBf16 — downcast fp32 → bf16, round-to-nearest-even.
//
// A cast is inherently mixed-precision, so the bit ops are hardcoded (see
// CastBf16ToF32 for the rationale). The bf16 write is PAIR-PACKED: one thread
// handles two consecutive elements and emits a single 32-bit store, so adjacent
// threads never race on the read-modify-write of a shared 32-bit word (the
// hazard in Storage.slang's non-atomic bf16 OaStore).
//
// Bindless: Stream.cpp prepends src_idx, dst_idx ahead of the user push.

struct PushConstants {
	uint src_idx;   // fp32 input  (prepended)
	uint dst_idx;   // bf16 output (prepended)
	uint count;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

// f32 → bf16 bits, round-to-nearest-even (halves the truncation bias).
uint F32ToBf16Bits(float val) {
	uint b = asuint(val);
	// Keep NaN a NaN (rounding could otherwise carry a NaN mantissa to inf).
	if ((b & 0x7FFFFFFFu) > 0x7F800000u) return (b >> 16u) | 0x0040u;
	uint rounding = 0x7FFFu + ((b >> 16u) & 1u);
	return (b + rounding) >> 16u;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint pair = tid.x;
	uint e0 = pair * 2u;
	if (e0 >= pc.count) return;

	uint h0 = F32ToBf16Bits(asfloat(heap[pc.src_idx].Load(e0 * 4u)));

	uint h1 = 0u;
	uint e1 = e0 + 1u;
	if (e1 < pc.count) {
		h1 = F32ToBf16Bits(asfloat(heap[pc.src_idx].Load(e1 * 4u)));
	}

	// One 32-bit store per element pair → no cross-thread RMW race on the word.
	heap[pc.dst_idx].Store(pair * 4u, h0 | (h1 << 16u));
}
