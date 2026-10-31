// Donor: oacpp/source/cpp/lib/oa/ui/shader/compute/drawCanvasGrid.slang
// Adaptation: spelling/indentation and explicit RGBA8 storage format only.
// Unified decimal viewport/chart/canvas grid over one clipped rectangle.

struct PushConstants {
	uint dst_idx;
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
	uint flags; // background, axes, grid tiers, guide fade, background dither
	int clip_x;
	int clip_y;
	uint clip_w;
	uint clip_h;
};

[[vk::push_constant]] PushConstants pc;
[[vk::image_format("rgba8")]]
[[vk::binding(1, 0)]] RWTexture2D<float4> images[];

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

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	if (tid.x >= pc.dst_w || tid.y >= pc.dst_h) return;
	int dst_x = pc.dst_x + int(tid.x);
	int dst_y = pc.dst_y + int(tid.y);
	if (dst_x < pc.clip_x || dst_y < pc.clip_y
		|| uint(dst_x - pc.clip_x) >= pc.clip_w
		|| uint(dst_y - pc.clip_y) >= pc.clip_h) return;

	uint image_width;
	uint image_height;
	images[pc.dst_idx].GetDimensions(image_width, image_height);
	if (dst_x < 0 || dst_y < 0
		|| dst_x >= int(image_width) || dst_y >= int(image_height)) return;

	const uint2 coordinate = uint2(uint(dst_x), uint(dst_y));
	const float2 pixel = float2(float(dst_x), float(dst_y)) + 0.5;
	const float2 relative = pixel - float2(pc.origin_x, pc.origin_y);
	float fade = 1.0;
	if ((pc.flags & 8U) != 0U) {
		fade = 1.0 - smoothstep(
			pc.fade_inner_radius,
			pc.fade_outer_radius,
			length(pixel - float2(pc.fade_center_x, pc.fade_center_y)));
	}
	float4 color = images[pc.dst_idx][coordinate];
	if ((pc.flags & 1U) != 0U) {
		const float fraction = pc.dst_h <= 1U
			? 0.0 : float(tid.y) / float(pc.dst_h - 1U);
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
	images[pc.dst_idx][coordinate] = color;
}
