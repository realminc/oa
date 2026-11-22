// UI raster anti-aliased filled screen-space triangle.
//
// Raster port of draw_triangle.slang. The scissor covers the element bounds;
// the fragment returns premultiplied alpha for the fixed blend.

struct PushConstants {
	float x0;
	float y0;
	float x1;
	float y1;
	float x2;
	float y2;
	uint rgba;
};

[[vk::push_constant]] PushConstants pc;

float edge(float2 begin, float2 end, float2 point) {
	float2 segment = end - begin;
	float2 offset = point - begin;
	return segment.x * offset.y - segment.y * offset.x;
}

float signed_edge_distance(
	float2 begin,
	float2 end,
	float2 point,
	float winding)
{
	float2 segment = end - begin;
	return winding * edge(begin, end, point)
		/ max(length(segment), 1.0e-8);
}

float4 unpack_rgba(uint rgba) {
	return float4(
		float((rgba >> 24) & 0xff) / 255.0,
		float((rgba >> 16) & 0xff) / 255.0,
		float((rgba >> 8) & 0xff) / 255.0,
		float(rgba & 0xff) / 255.0);
}

[shader("fragment")]
float4 main(float4 position : SV_Position) : SV_Target0 {
	float2 p0 = float2(pc.x0, pc.y0);
	float2 p1 = float2(pc.x1, pc.y1);
	float2 p2 = float2(pc.x2, pc.y2);
	float area = edge(p0, p1, p2);
	if (abs(area) <= 1.0e-8) discard;
	float winding = area < 0.0 ? -1.0 : 1.0;
	float2 pixel = position.xy;
	float distance_to_edge = min(
		signed_edge_distance(p0, p1, pixel, winding),
		min(
			signed_edge_distance(p1, p2, pixel, winding),
			signed_edge_distance(p2, p0, pixel, winding)));
	float coverage = saturate(distance_to_edge + 0.5);
	float4 src = unpack_rgba(pc.rgba);
	float alpha = src.a * coverage;
	return float4(src.rgb * alpha, alpha);
}
