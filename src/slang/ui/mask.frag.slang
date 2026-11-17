// UI raster premultiplied rounded viewport mask.
//
// Raster port of mask_rounded_viewport.slang. One fragment serves two draws
// selected by `mode`:
//   mode 0 — returns vec4(coverage); the pipeline's modulate blend multiplies
//            the attachment by coverage (dst * src).
//   mode 1 — returns premultiplied `outside * (1 - coverage)`; the pipeline's
//            additive blend adds it.
// Together they reproduce the compute path's
// `current * coverage + outside_premultiplied * (1 - coverage)` exactly.

struct PushConstants {
	int dst_x;
	int dst_y;
	uint dst_w;
	uint dst_h;
	float corner_radius;
	uint outside_rgba;
	uint mode;
};

[[vk::push_constant]] PushConstants pc;

float rounded_viewport_coverage(float2 local) {
	float radius = min(
		max(pc.corner_radius, 0.0),
		0.5 * float(min(pc.dst_w, pc.dst_h)));
	if (radius <= 0.0) return 1.0;
	float2 half_size = 0.5 * float2(pc.dst_w, pc.dst_h);
	float2 q = abs(local - half_size) - (half_size - radius);
	float distance = length(max(q, float2(0.0, 0.0)))
		+ min(max(q.x, q.y), 0.0) - radius;
	return saturate(0.5 - distance);
}

float4 unpack_premultiplied_rgba(uint rgba) {
	float4 color = float4(
		float((rgba >> 24) & 0xff) / 255.0,
		float((rgba >> 16) & 0xff) / 255.0,
		float((rgba >> 8) & 0xff) / 255.0,
		float(rgba & 0xff) / 255.0);
	return float4(color.rgb * color.a, color.a);
}

[shader("fragment")]
float4 main(float4 position : SV_Position) : SV_Target0 {
	float2 local = position.xy - float2(float(pc.dst_x), float(pc.dst_y));
	float coverage = rounded_viewport_coverage(local);
	if (pc.mode == 0u) {
		return float4(coverage, coverage, coverage, coverage);
	}
	return unpack_premultiplied_rgba(pc.outside_rgba) * (1.0 - coverage);
}
