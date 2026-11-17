// UI raster filled rectangle — alpha-blended, optional rounded corners.
//
// Raster port of draw_rect.slang. The scissor replaces the per-pixel clip and
// image-bounds tests; the fragment returns premultiplied alpha so the fixed
// pipeline blend produces the compute path's source-over result bit-exactly.
//
// Push layout (24 bytes):
//   offset  0 : int2  dst_origin
//   offset  8 : uint2 dst_size
//   offset 16 : uint  rgba
//   offset 20 : float corner_radius

#include "compositor_common.slang"

struct PushConstants {
	int dst_x;
	int dst_y;
	uint dst_w;
	uint dst_h;
	uint rgba;
	float corner_radius;
};

[[vk::push_constant]] PushConstants pc;

float rounded_rect_coverage(float2 local) {
	float radius = min(
		max(pc.corner_radius, 0.0),
		0.5 * float(min(pc.dst_w, pc.dst_h)));
	if (radius <= 0.0) return 1.0;
	float2 half_size = 0.5 * float2(float(pc.dst_w), float(pc.dst_h));
	float2 q = abs(local - half_size) - (half_size - radius);
	float distance = length(max(q, float2(0.0, 0.0)))
		+ min(max(q.x, q.y), 0.0) - radius;
	return saturate(0.5 - distance);
}

[shader("fragment")]
float4 main(float4 position : SV_Position) : SV_Target0 {
	// Pixel-center coordinates match the compute path's `float(x) + 0.5`.
	float2 local = position.xy - float2(float(pc.dst_x), float(pc.dst_y));
	float coverage = rounded_rect_coverage(local);
	float4 src = uiUnpackRgba(pc.rgba);
	float alpha = src.a * coverage;
	return float4(src.rgb * alpha, alpha);
}
