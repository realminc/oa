// MatrixToRgba8.slang — [1,C,H,W] floating matrix to packed RGBA8.
//
// C=1 produces grayscale, C=3 adds opaque alpha, C=4 preserves alpha.
// Values are clamped to [0,1]. One uint output element is one RGBA8 pixel.

import Storage;

struct PushConstants {
	uint in_idx;
	uint out_idx;
	uint channels;
	uint height;
	uint width;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[ForceInline]
uint ToByte(float value) {
	return uint(saturate(value) * 255.0f + 0.5f);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint pixel = tid.x;
	uint pixel_count = pc.width * pc.height;
	if (pixel >= pixel_count) return;

	float r = OaLoad(heap[pc.in_idx], pixel);
	float g = r;
	float b = r;
	float a = 1.0f;
	if (pc.channels >= 3u) {
		g = OaLoad(heap[pc.in_idx], pixel_count + pixel);
		b = OaLoad(heap[pc.in_idx], pixel_count * 2u + pixel);
	}
	if (pc.channels == 4u) {
		a = OaLoad(heap[pc.in_idx], pixel_count * 3u + pixel);
	}

	uint packed = ToByte(r)
		| (ToByte(g) << 8u)
		| (ToByte(b) << 16u)
		| (ToByte(a) << 24u);
	heap[pc.out_idx].Store(pixel * 4u, packed);
}
