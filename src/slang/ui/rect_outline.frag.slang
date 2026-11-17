// UI raster rectangle outline — alpha-blended border, optional rounded corners.
//
// Raster port of draw_rect_outline.slang. The scissor replaces the per-pixel
// clip and image-bounds tests; output is premultiplied for the fixed blend.

struct PushConstants {
	int dst_x;
	int dst_y;
	uint dst_w;
	uint dst_h;
	uint thickness;
	uint rgba;
	float corner_radius;
};

[[vk::push_constant]] PushConstants pc;

float rounded_rect_distance(float2 local, float2 half_size, float radius) {
	float2 q = abs(local) - (half_size - radius);
	return length(max(q, float2(0.0, 0.0)))
		+ min(max(q.x, q.y), 0.0) - radius;
}

float4 unpack_rgba(uint rgba) {
	return float4(
		float((rgba >> 24) & 0xffu) / 255.0,
		float((rgba >> 16) & 0xffu) / 255.0,
		float((rgba >> 8) & 0xffu) / 255.0,
		float(rgba & 0xffu) / 255.0);
}

[shader("fragment")]
float4 main(float4 position : SV_Position) : SV_Target0 {
	float2 local = position.xy - float2(float(pc.dst_x), float(pc.dst_y));
	uint thickness = max(pc.thickness, 1u);
	float coverage = 1.0;
	float radius = min(
		max(pc.corner_radius, 0.0),
		0.5 * float(min(pc.dst_w, pc.dst_h)));
	if (radius <= 0.0) {
		bool border = local.x - 0.5 < float(thickness)
			|| local.y - 0.5 < float(thickness)
			|| local.x - 0.5 >= float(pc.dst_w - min(thickness, pc.dst_w))
			|| local.y - 0.5 >= float(pc.dst_h - min(thickness, pc.dst_h));
		if (!border) discard;
	} else {
		float2 half_size = 0.5 * float2(float(pc.dst_w), float(pc.dst_h));
		float outer_coverage = saturate(
			0.5 - rounded_rect_distance(local - half_size, half_size, radius));
		float inset = min(float(thickness), min(half_size.x, half_size.y));
		float2 inner_half_size = half_size - inset;
		float inner_coverage = 0.0;
		if (inner_half_size.x > 0.0 && inner_half_size.y > 0.0) {
			float inner_radius = max(0.0, radius - inset);
			inner_coverage = saturate(
				0.5 - rounded_rect_distance(
					local - half_size, inner_half_size, inner_radius));
		}
		coverage = saturate(outer_coverage - inner_coverage);
		if (coverage <= 0.0) discard;
	}
	float4 src = unpack_rgba(pc.rgba);
	float alpha = src.a * coverage;
	return float4(src.rgb * alpha, alpha);
}
