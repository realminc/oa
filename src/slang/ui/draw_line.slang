// DrawLine.slang — anti-aliased screen-space line segment on compose image.
//
// Donor: oacpp/source/cpp/lib/oa/ui/shader/compute/drawLine.slang
// Adaptation: exact donor shader body.
//
// Dispatch: (ceil(bounds_w / 8), ceil(bounds_h / 8), 1)

#include "compositor_common.slang"

struct PushConstants {
	uint  dst_idx;
	float x0;
	float y0;
	float x1;
	float y1;
	float thickness;
	uint  rgba;
	uint  bounds_x;
	uint  bounds_y;
	uint  bounds_w;
	uint  bounds_h;
}

[[vk::push_constant]] PushConstants pc;
[[vk::binding(1, 0)]] RWTexture2D<float4> images[];

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	if (tid.x >= pc.bounds_w || tid.y >= pc.bounds_h) return;
	float2 pixel = float2(float(pc.bounds_x + tid.x), float(pc.bounds_y + tid.y)) + 0.5;
	float2 begin = float2(pc.x0, pc.y0);
	float2 segment = float2(pc.x1, pc.y1) - begin;
	float lengthSquared = max(dot(segment, segment), 1.0e-8);
	float along = saturate(dot(pixel - begin, segment) / lengthSquared);
	float distanceToLine = length(pixel - (begin + along * segment));
	float coverage = 1.0 - smoothstep(
		max(0.0, pc.thickness * 0.5 - 1.0),
		pc.thickness * 0.5 + 1.0,
		distanceToLine);
	if (coverage <= 0.0) return;
	uint imageWidth;
	uint imageHeight;
	images[pc.dst_idx].GetDimensions(imageWidth, imageHeight);
	uint2 coordinate = uint2(pc.bounds_x + tid.x, pc.bounds_y + tid.y);
	if (coordinate.x >= imageWidth || coordinate.y >= imageHeight) return;
	float4 src = uiUnpackRgba(pc.rgba);
	src.a *= coverage;
	float4 dst = images[pc.dst_idx][coordinate];
	images[pc.dst_idx][coordinate] = uiSourceOver(src, dst);
}
