// MatrixToRgba8.slang — semantic floating-point image to packed RGBA8.
//
// Layout and format use the stable oa::ImageLayout/oa::ImageFormat ordinal values.
// Values are clamped to [0,1]. One uint output element is one RGBA8 pixel.

import storage;

struct PushConstants {
	uint in_idx;
	uint out_idx;
	uint channels;
	uint height;
	uint width;
	uint layout;
	uint format;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[ForceInline]
uint ToByte(float value) {
	return uint(saturate(value) * 255.0f + 0.5f);
}

[ForceInline]
float LoadChannel(uint pixel, uint channel) {
	uint index = pixel;
	// NCHW (0) and CHW (2) are planar. NHWC (1) and HWC (3) are
	// interleaved. HW (4) has one implicit grayscale channel.
	if (pc.layout == 0u || pc.layout == 2u) {
		index += channel * pc.width * pc.height;
	} else if (pc.layout != 4u) {
		index = pixel * pc.channels + channel;
	}
	return oaLoad(heap[pc.in_idx], index);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint pixel = tid.x;
	uint pixel_count = pc.width * pc.height;
	if (pixel >= pixel_count) return;

	float r = LoadChannel(pixel, 0u);
	float g = r;
	float b = r;
	float a = 1.0f;
	if (pc.format == 1u) { // GrayAlpha
		a = LoadChannel(pixel, 1u);
	} else if (pc.format >= 2u) {
		g = LoadChannel(pixel, 1u);
		if (pc.format == 4u || pc.format == 5u) { // Bgr / Bgra
			b = r;
			r = LoadChannel(pixel, 2u);
		} else { // Rgb / Rgba
			b = LoadChannel(pixel, 2u);
		}
		if (pc.format == 3u || pc.format == 5u) {
			a = LoadChannel(pixel, 3u);
		}
	}

	uint packed = ToByte(r)
		| (ToByte(g) << 8u)
		| (ToByte(b) << 16u)
		| (ToByte(a) << 24u);
	heap[pc.out_idx].Store(pixel * 4u, packed);
}
