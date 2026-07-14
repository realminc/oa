// CastBf16ToF32 — upcast bf16 → fp32.
//
// A cast is inherently mixed-precision (src dtype ≠ dst dtype), so it CANNOT use
// the branched Storage helpers (one DTYPE spec constant can't describe two
// dtypes). Both directions are hardcoded here. One thread per element; the fp32
// output is naturally aligned, so no read-modify-write race (unlike bf16 stores).
//
// Bindless: Stream.cpp prepends src_idx, dst_idx ahead of the user push.

struct PushConstants {
	uint src_idx;   // bf16 input  (prepended)
	uint dst_idx;   // fp32 output (prepended)
	uint count;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint idx = tid.x;
	if (idx >= pc.count) return;

	uint byte_off = idx * 2u;
	uint word = heap[pc.src_idx].Load(byte_off & ~3u);
	uint shift = (byte_off & 2u) * 8u;
	uint bf16 = (word >> shift) & 0xFFFFu;
	float v = asfloat(bf16 << 16u);

	heap[pc.dst_idx].Store(idx * 4u, asuint(v));
}
