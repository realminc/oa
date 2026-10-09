// DrawRect.slang — alpha-blended filled rectangle on the compose storage image.
//
// Donor: oacpp/source/cpp/lib/oa/ui/shader/compute/drawRect.slang
// Adaptation: exact donor shader body.
//
// Dispatch: (ceil(dst_w / 8), ceil(dst_h / 8), 1)

#include "compositor_common.slang"

struct PushConstants {
	uint dst_idx;
	int  dst_x;
	int  dst_y;
	uint dst_w;
	uint dst_h;
	uint rgba;
	int  clip_x;
	int  clip_y;
	uint clip_w;
	uint clip_h;
	float corner_radius;
}

[[vk::push_constant]] PushConstants pc;
[[vk::binding(1, 0)]] RWTexture2D<float4> images[];

float roundedRectCoverage(uint x, uint y) {
	float radius = min(
		max(pc.corner_radius, 0.0),
		0.5 * float(min(pc.dst_w, pc.dst_h)));
	if (radius <= 0.0) return 1.0;
	float2 halfSize = 0.5 * float2(float(pc.dst_w), float(pc.dst_h));
	float2 local = float2(float(x), float(y)) + 0.5 - halfSize;
	float2 q = abs(local) - (halfSize - radius);
	float distance = length(max(q, float2(0.0, 0.0)))
		+ min(max(q.x, q.y), 0.0) - radius;
	return saturate(0.5 - distance);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
	uint x = gid.x * 8 + lid.x;
	uint y = gid.y * 8 + lid.y;
	if (x >= pc.dst_w || y >= pc.dst_h) return;

	int dstX = pc.dst_x + int(x);
	int dstY = pc.dst_y + int(y);
	if (dstX < pc.clip_x || dstY < pc.clip_y
		|| uint(dstX - pc.clip_x) >= pc.clip_w
		|| uint(dstY - pc.clip_y) >= pc.clip_h) return;
	uint imageWidth;
	uint imageHeight;
	images[pc.dst_idx].GetDimensions(imageWidth, imageHeight);
	if (dstX < 0 || dstY < 0
		|| dstX >= int(imageWidth) || dstY >= int(imageHeight)) return;

	float coverage = roundedRectCoverage(x, y);
	if (coverage <= 0.0) return;

	float4 src = uiUnpackRgba(pc.rgba);
	src.a *= coverage;
	uint2 dstCoord = uint2(uint(dstX), uint(dstY));
	float4 dst = images[pc.dst_idx][dstCoord];
	images[pc.dst_idx][dstCoord] = uiSourceOver(src, dst);
}
