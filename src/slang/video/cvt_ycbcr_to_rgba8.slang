// Donor: oacpp/source/cpp/lib/oa/vision/shader/compute/conversion/
// cvtNv12YcbcrToRgba.slang. Mechanical ABI adaptation: packed RGBA8 buffer
// output instead of a storage image; hardware sample/conversion is unchanged.
struct PushConstants {
	uint out_idx;
	uint width;
	uint height;
	uint coded_width;
	uint coded_height;
};
[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];
[[vk::binding(0, 1)]] Sampler2D<float4> ycbcr_image;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
	uint x = gid.x * 16 + lid.x;
	uint y = gid.y * 16 + lid.y;
	if (x >= pc.width || y >= pc.height) return;
	float2 uv = (float2(x, y) + 0.5f) / float2(pc.coded_width, pc.coded_height);
	float4 color = ycbcr_image.SampleLevel(uv, 0);
	uint3 rgb = uint3(round(saturate(color.rgb) * 255.0f));
	heap[pc.out_idx].Store((y * pc.width + x) * 4u,
		rgb.r | (rgb.g << 8u) | (rgb.b << 16u) | (255u << 24u));
}
