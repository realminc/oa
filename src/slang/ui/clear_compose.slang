// ClearCompose.slang — fill the compose storage image with one RGBA color.
//
// Donor: oacpp/source/cpp/lib/oa/ui/shader/compute/clearRgba8.slang
// Adaptation: operates on a storage image (float4) rather than a byte buffer.
//
// Dispatch: (ceil(width / 8), ceil(height / 8), 1)

struct PushConstants {
	uint dst_idx;
	uint dst_w;
	uint dst_h;
	uint rgba;
}

[[vk::push_constant]] PushConstants pc;
[[vk::binding(1, 0)]] RWTexture2D<float4> images[];

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	if (tid.x >= pc.dst_w || tid.y >= pc.dst_h) return;
	uint imageWidth;
	uint imageHeight;
	images[pc.dst_idx].GetDimensions(imageWidth, imageHeight);
	if (tid.x >= imageWidth || tid.y >= imageHeight) return;
	images[pc.dst_idx][tid.xy] = float4(
		float((pc.rgba >> 24) & 0xffu) / 255.0,
		float((pc.rgba >> 16) & 0xffu) / 255.0,
		float((pc.rgba >> 8) & 0xffu) / 255.0,
		float(pc.rgba & 0xffu) / 255.0);
}
