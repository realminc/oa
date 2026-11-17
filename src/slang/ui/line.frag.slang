// UI raster anti-aliased screen-space line segment.
//
// Raster port of draw_line.slang. The scissor covers the segment bounds; the
// fragment returns premultiplied alpha for the fixed blend.

struct PushConstants {
	float x0;
	float y0;
	float x1;
	float y1;
	float thickness;
	uint rgba;
};

[[vk::push_constant]] PushConstants pc;

float4 unpack_rgba(uint rgba) {
	return float4(
		float((rgba >> 24) & 0xffu) / 255.0,
		float((rgba >> 16) & 0xffu) / 255.0,
		float((rgba >> 8) & 0xffu) / 255.0,
		float(rgba & 0xffu) / 255.0);
}

[shader("fragment")]
float4 main(float4 position : SV_Position) : SV_Target0 {
	float2 pixel = position.xy;
	float2 begin = float2(pc.x0, pc.y0);
	float2 segment = float2(pc.x1, pc.y1) - begin;
	float length_squared = max(dot(segment, segment), 1.0e-8);
	float along = saturate(dot(pixel - begin, segment) / length_squared);
	float distance_to_line = length(pixel - (begin + along * segment));
	float coverage = 1.0 - smoothstep(
		max(0.0, pc.thickness * 0.5 - 1.0),
		pc.thickness * 0.5 + 1.0,
		distance_to_line);
	float4 src = unpack_rgba(pc.rgba);
	float alpha = src.a * coverage;
	return float4(src.rgb * alpha, alpha);
}
