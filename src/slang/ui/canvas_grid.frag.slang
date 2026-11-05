// CanvasGrid fragment — raster port of ui/draw_canvas_grid.slang.
//
// The compute shader composites each grid layer directly onto the compose
// image. Here the fragment folds the same layers over transparent black and
// the blend unit composites the result over the attachment; source-over is
// associative so the output is identical. The pipeline's scissor replaces the
// per-pixel clip test.
//
// Push layout (100 bytes):
//   offset  0 : int2   dst_origin           element rect origin (pixels)
//   offset  8 : uint2  dst_size             element rect size (pixels)
//   offset 16 : float2 origin               grid line origin (pixels)
//   offset 24 : float2 minor_spacing
//   offset 32 : uint   major_every
//   offset 36 : uint   super_major_every
//   offset 40 : float4 line_thicknesses     minor/major/super_major/axis
//   offset 56 : uint   background_top_rgba
//   offset 60 : uint   background_bottom_rgba
//   offset 64 : uint   minor_rgba
//   offset 68 : uint   major_rgba
//   offset 72 : uint   super_major_rgba
//   offset 76 : uint   axis_rgba
//   offset 80 : float2 fade_center
//   offset 88 : float2 fade_radii           inner, outer
//   offset 96 : uint   flags — 1 background · 2 axes · 4 grid · 8 fade · 16 dither

struct PushConstants {
	int dst_x;
	int dst_y;
	uint dst_w;
	uint dst_h;
	float origin_x;
	float origin_y;
	float minor_spacing_x;
	float minor_spacing_y;
	uint major_every;
	uint super_major_every;
	float minor_thickness;
	float major_thickness;
	float super_major_thickness;
	float axis_thickness;
	uint background_top_rgba;
	uint background_bottom_rgba;
	uint minor_rgba;
	uint major_rgba;
	uint super_major_rgba;
	uint axis_rgba;
	float fade_center_x;
	float fade_center_y;
	float fade_inner_radius;
	float fade_outer_radius;
	uint flags;
};

[[vk::push_constant]] PushConstants pc;

float4 unpack_rgba(uint rgba) {
	return float4(
		float((rgba >> 24) & 0xff) / 255.0,
		float((rgba >> 16) & 0xff) / 255.0,
		float((rgba >> 8) & 0xff) / 255.0,
		float(rgba & 0xff) / 255.0);
}

float4 blend_over(float4 dst, float4 src, float coverage) {
	src.a *= coverage;
	return float4(
		src.rgb * src.a + dst.rgb * (1.0 - src.a),
		src.a + dst.a * (1.0 - src.a));
}

float distance_to_periodic_line(float value, float spacing) {
	return abs(frac(value / spacing + 0.5) - 0.5) * spacing;
}

float line_coverage(float distance_to_line, float thickness) {
	return 1.0 - smoothstep(
		max(0.0, thickness * 0.5 - 0.75),
		thickness * 0.5 + 0.75,
		distance_to_line);
}

float spatial_dither(uint2 coordinate) {
	uint value = coordinate.x + coordinate.y * 65537U;
	value = (value ^ 61U) ^ (value >> 16U);
	value *= 9U;
	value ^= value >> 4U;
	value *= 0x27d4eb2dU;
	value ^= value >> 15U;
	return float(value & 255U) / 255.0 - 0.5;
}

[shader("fragment")]
float4 main(float4 position : SV_Position) : SV_Target0 {
	// sv_position is the framebuffer pixel center — identical to the compute
	// path's `float(dst) + 0.5`.
	const uint2 coordinate = uint2(uint(position.x), uint(position.y));
	const float2 pixel = position.xy;
	const float2 relative = pixel - float2(pc.origin_x, pc.origin_y);
	float fade = 1.0;
	if ((pc.flags & 8U) != 0U) {
		fade = 1.0 - smoothstep(
			pc.fade_inner_radius,
			pc.fade_outer_radius,
			length(pixel - float2(pc.fade_center_x, pc.fade_center_y)));
	}
	float4 color = float4(0.0, 0.0, 0.0, 0.0);
	if ((pc.flags & 1U) != 0U) {
		// local_y must equal the compute path's integer tid.y.
		const float local_y = pixel.y - float(pc.dst_y) - 0.5;
		const float fraction = pc.dst_h <= 1U
			? 0.0 : local_y / float(pc.dst_h - 1U);
		float4 background = lerp(
			unpack_rgba(pc.background_top_rgba),
			unpack_rgba(pc.background_bottom_rgba),
			fraction);
		if ((pc.flags & 16U) != 0U) {
			background.rgb = saturate(
				background.rgb
					+ spatial_dither(coordinate) * (2.0 / 255.0));
		}
		color = blend_over(color, background, 1.0);
	}

	if ((pc.flags & 4U) != 0U) {
		const float minor_distance = min(
			distance_to_periodic_line(relative.x, pc.minor_spacing_x),
			distance_to_periodic_line(relative.y, pc.minor_spacing_y));
		color = blend_over(color, unpack_rgba(pc.minor_rgba),
			line_coverage(minor_distance, pc.minor_thickness) * fade);

		const float2 major_spacing = float2(
			pc.minor_spacing_x, pc.minor_spacing_y) * float(pc.major_every);
		const float major_distance = min(
			distance_to_periodic_line(relative.x, major_spacing.x),
			distance_to_periodic_line(relative.y, major_spacing.y));
		color = blend_over(color, unpack_rgba(pc.major_rgba),
			line_coverage(major_distance, pc.major_thickness) * fade);

		const float2 super_major_spacing = float2(
			pc.minor_spacing_x, pc.minor_spacing_y)
			* float(pc.super_major_every);
		const float super_major_distance = min(
			distance_to_periodic_line(relative.x, super_major_spacing.x),
			distance_to_periodic_line(relative.y, super_major_spacing.y));
		color = blend_over(color, unpack_rgba(pc.super_major_rgba),
			line_coverage(super_major_distance, pc.super_major_thickness) * fade);
	}

	if ((pc.flags & 2U) != 0U) {
		const float axis_distance = min(abs(relative.x), abs(relative.y));
		color = blend_over(color, unpack_rgba(pc.axis_rgba),
			line_coverage(axis_distance, pc.axis_thickness) * fade);
	}
	return color;
}
