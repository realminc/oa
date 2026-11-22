// UI raster line plot — per-pixel port of draw_plot_line.slang.
//
// The compute shader blended each column's segment contributions into the
// compose image in turn. A fragment cannot read-modify-write the attachment,
// so it folds the same contributions analytically: source-over of same-color
// sources composes to `1 - Π(1 - aᵢ)`, which yields an identical final
// premultiplied output for the fixed pipeline blend.
//
// Push layout (56 bytes):
//   offset  0 : uint  count
//   offset  4 : int2  dst_origin
//   offset 12 : uint2 dst_size
//   offset 20 : float x_min   offset 24 : float x_max
//   offset 28 : float y_min   offset 32 : float y_max
//   offset 36 : uint  rgba
//   offset 40 : uint  flags — bit 1 fill · bit 2 packed explicit X/Y
//                            · bit 3 band fill between two y blocks
//   offset 44 : uint  antialias_samples — admitted values: 1, 4, 8
//   offset 48 : float line_width

import storage;

struct PushConstants {
	uint count;
	int dst_x;
	int dst_y;
	uint dst_w;
	uint dst_h;
	float x_min;
	float x_max;
	float y_min;
	float y_max;
	uint rgba;
	uint flags;
	uint antialias_samples;
	float line_width;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] ByteAddressBuffer heap[1];

float4 unpack_rgba(uint rgba) {
	return float4(
		float((rgba >> 24) & 0xffu) / 255.0,
		float((rgba >> 16) & 0xffu) / 255.0,
		float((rgba >> 8) & 0xffu) / 255.0,
		float(rgba & 0xffu) / 255.0);
}

float sample_at(float position) {
	return sample_block(0u, position);
}

// Interpolated read of one `count`-sample block starting at `offset`.
float sample_block(uint offset, float position) {
	if (pc.count <= 1) return oaLoad(heap[0], offset);
	float index = clamp(position, 0.0, 1.0) * float(pc.count - 1);
	uint lo = uint(floor(index));
	uint hi = min(pc.count - 1, lo + 1);
	return lerp(
		oaLoad(heap[0], offset + lo),
		oaLoad(heap[0], offset + hi),
		index - float(lo));
}

float plot_y(float value) {
	float normalized = saturate((value - pc.y_min) / (pc.y_max - pc.y_min));
	return (1.0 - normalized) * float(pc.dst_h - 1);
}

float plot_x(float value) {
	return saturate((value - pc.x_min) / (pc.x_max - pc.x_min))
		* float(pc.dst_w - 1);
}

float distance_to_segment(float2 point, float2 begin, float2 end) {
	float2 edge = end - begin;
	float length_squared = dot(edge, edge);
	if (length_squared <= 1.0e-8) return length(point - begin);
	float fraction = saturate(dot(point - begin, edge) / length_squared);
	return length(point - (begin + fraction * edge));
}

float2 sample_offset(uint index, uint count) {
	if (count <= 1u) return float2(0.5, 0.5);
	if (count <= 4u) {
		return float2(
			(index & 1u) == 0u ? 0.25 : 0.75,
			(index & 2u) == 0u ? 0.25 : 0.75);
	}
	switch (index & 7u) {
		case 0u: return float2(0.125, 0.375);
		case 1u: return float2(0.375, 0.875);
		case 2u: return float2(0.625, 0.125);
		case 3u: return float2(0.875, 0.625);
		case 4u: return float2(0.125, 0.875);
		case 5u: return float2(0.375, 0.125);
		case 6u: return float2(0.625, 0.625);
		default: return float2(0.875, 0.375);
	}
}

float line_coverage(int2 pixel, float2 begin, float2 end) {
	uint count = pc.antialias_samples == 8u
		? 8u : pc.antialias_samples == 1u ? 1u : 4u;
	float covered = 0.0;
	float half_width = max(0.25, pc.line_width * 0.5);
	for (uint sample = 0u; sample < 8u; ++sample) {
		if (sample >= count) break;
		float distance = distance_to_segment(
			float2(pixel) + sample_offset(sample, count), begin, end);
		covered += 1.0 - smoothstep(
			max(0.0, half_width - 0.45), half_width + 0.45, distance);
	}
	return covered / float(count);
}

// Fold one segment's contribution into the running product of uncovered
// alpha, matching the compute path's column loop and x-rejection.
void fold_segment(inout float remaining, int2 pixel, float2 begin, float2 end, float alpha) {
	float radius = max(1.0, pc.line_width * 0.5 + 1.0);
	if (float(pixel.x) + 1.0 < min(begin.x, end.x) - radius
		|| float(pixel.x) > max(begin.x, end.x) + radius) {
		return;
	}
	int minimum_y = max(0, int(floor(min(begin.y, end.y) - radius)));
	int maximum_y = min(int(pc.dst_h) - 1,
		int(ceil(max(begin.y, end.y) + radius)));
	if (pixel.y < minimum_y || pixel.y > maximum_y) return;
	float coverage = line_coverage(pixel, begin, end);
	remaining *= 1.0 - alpha * coverage;
}

[shader("fragment")]
float4 main(float4 position : SV_Position) : SV_Target0 {
	uint x = uint(int(position.x) - pc.dst_x);
	int2 pixel = int2(int(position.x) - pc.dst_x, int(position.y) - pc.dst_y);
	if (x >= pc.dst_w || pc.count == 0 || pc.dst_h == 0) discard;

	float4 color = unpack_rgba(pc.rgba);
	float remaining = 1.0;

	if ((pc.flags & 4u) != 0u) {
		if (pc.x_max <= pc.x_min || pc.y_max <= pc.y_min) discard;
		if (pc.count == 1) {
			float data_x = oaLoad(heap[0], 0);
			float data_y = oaLoad(heap[0], 1);
			if (isfinite(data_x) && isfinite(data_y)) {
				float2 point = float2(plot_x(data_x), plot_y(data_y));
				fold_segment(remaining, pixel, point, point, color.a);
			}
		} else {
			for (uint segment = 0; segment + 1 < pc.count; ++segment) {
				float data_x0 = oaLoad(heap[0], segment);
				float data_x1 = oaLoad(heap[0], segment + 1);
				float data_y0 = oaLoad(heap[0], pc.count + segment);
				float data_y1 = oaLoad(heap[0], pc.count + segment + 1);
				if (!isfinite(data_x0) || !isfinite(data_x1)
					|| !isfinite(data_y0) || !isfinite(data_y1)) continue;
				float screen_x0 = plot_x(data_x0);
				float screen_x1 = plot_x(data_x1);
				if (abs(screen_x1 - screen_x0) >= 0.5) {
					float minimum_x = min(screen_x0, screen_x1);
					float maximum_x = max(screen_x0, screen_x1);
					if (float(pixel.x) + 1.0 < minimum_x - pc.line_width
						|| float(pixel.x) > maximum_x + pc.line_width) continue;
				}
				fold_segment(
					remaining,
					pixel,
					float2(screen_x0, plot_y(data_y0)),
					float2(screen_x1, plot_y(data_y1)),
					color.a);
			}
		}
	} else if ((pc.flags & 8u) != 0u) {
		// Band fill between two `count`-sample curves: y_lo at block 0,
		// y_hi at block `count`. A NaN on either boundary drops the column.
		float position_x = pc.dst_w <= 1 ? 0.0 : float(x) / float(pc.dst_w - 1);
		float lo = sample_block(0u, position_x);
		float hi = sample_block(pc.count, position_x);
		if (isfinite(lo) && isfinite(hi)) {
			float edge_top = min(plot_y(lo), plot_y(hi));
			float edge_bottom = max(plot_y(lo), plot_y(hi));
			float coverage = smoothstep(edge_top - 0.5, edge_top + 0.5, float(pixel.y))
				* (1.0 - smoothstep(edge_bottom - 0.5, edge_bottom + 0.5, float(pixel.y)));
			remaining *= 1.0 - color.a * coverage;
		}
	} else {
		float position_x = pc.dst_w <= 1 ? 0.0 : float(x) / float(pc.dst_w - 1);
		float previous_x = pc.dst_w <= 1 || x == 0
			? position_x : float(x - 1) / float(pc.dst_w - 1);
		float y0 = plot_y(sample_at(previous_x));
		float y1 = plot_y(sample_at(position_x));

		if ((pc.flags & 2u) != 0u
			&& pixel.y >= int(ceil(max(y0, y1)))
			&& pixel.y < int(pc.dst_h)) {
			remaining *= 1.0 - color.a * 0.16;
		}
		fold_segment(
			remaining,
			pixel,
			float2(float(x == 0u ? 0u : x - 1u), y0),
			float2(float(x), y1),
			color.a);
	}

	float alpha = 1.0 - remaining;
	return float4(color.rgb * alpha, alpha);
}
