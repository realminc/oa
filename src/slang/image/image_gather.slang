// Shared gather math for the tile-IR image geometric family (OA-CC-33).
//
// Source coordinates are continuous pixel positions in the source plane.
// `base` is the batch-channel plane offset; `width`/`height` the source
// extent. The fetch clamps coordinates to the plane's pixel centers and
// lerps the four covering texels — the same math as the resize_bilinear
// donor kernel.

import storage;

// Clamped bilinear fetch of one f32 element from a packed image plane.
[ForceInline]
export float image_bilinear_f32(
	RWByteAddressBuffer buffer,
	uint base,
	float xf,
	float yf,
	uint width,
	uint height
) {
	xf = clamp(xf, 0.0f, float(width - 1u));
	yf = clamp(yf, 0.0f, float(height - 1u));
	uint x0 = min((uint)floor(xf), width - 1u);
	uint y0 = min((uint)floor(yf), height - 1u);
	uint x1 = min(x0 + 1u, width - 1u);
	uint y1 = min(y0 + 1u, height - 1u);
	float x_weight = xf - float(x0);
	float y_weight = yf - float(y0);
	float top_left = load_f32(buffer, base + y0 * width + x0);
	float top_right = load_f32(buffer, base + y0 * width + x1);
	float bottom_left = load_f32(buffer, base + y1 * width + x0);
	float bottom_right = load_f32(buffer, base + y1 * width + x1);
	float top = lerp(top_left, top_right, x_weight);
	float bottom = lerp(bottom_left, bottom_right, x_weight);
	return lerp(top, bottom, y_weight);
}

// Border-mode coordinate folding shared by pad, remap, and the warps:
// 0 constant (out of range), 1 replicate, 2 mirror, 3 reflect, 4 wrap.
[ForceInline]
export int image_border_coordinate(int coordinate, uint extent, uint mode) {
	if (coordinate >= 0 && coordinate < int(extent)) return coordinate;
	if (mode == 0u) return -1;
	if (mode == 1u) return clamp(coordinate, 0, int(extent) - 1);
	if (extent == 1u) return 0;
	int period = mode == 2u ? int(2u * extent) : int(2u * (extent - 1u));
	if (mode == 4u) period = int(extent);
	int folded = coordinate % period;
	if (folded < 0) folded += period;
	if (mode == 4u || folded < int(extent)) return folded;
	return mode == 2u ? period - 1 - folded : period - folded;
}

// One bordered texel fetch — negative folded coordinates take the fill.
[ForceInline]
export float image_sample_bordered(
	RWByteAddressBuffer buffer,
	uint base,
	int y,
	int x,
	uint width,
	uint height,
	uint mode,
	float border_value
) {
	int source_x = image_border_coordinate(x, width, mode);
	int source_y = image_border_coordinate(y, height, mode);
	if (source_x < 0 || source_y < 0) return border_value;
	return load_f32(buffer, base + uint(source_y) * width + uint(source_x));
}

// Bordered gather matching the remap/warp `sample_image` donors:
// interpolation 0 rounds to the nearest texel, otherwise bilinear over
// the four covering texels, each through the border fold.
[ForceInline]
export float image_gather_bordered(
	RWByteAddressBuffer buffer,
	uint base,
	float y,
	float x,
	uint width,
	uint height,
	uint interp,
	uint mode,
	float border_value
) {
	if (interp == 0u) {
		return image_sample_bordered(
			buffer, base, int(floor(y + 0.5f)), int(floor(x + 0.5f)),
			width, height, mode, border_value);
	}
	int y0 = int(floor(y)), x0 = int(floor(x));
	float fy = y - float(y0), fx = x - float(x0);
	return lerp(
		lerp(image_sample_bordered(buffer, base, y0, x0, width, height, mode, border_value),
			image_sample_bordered(buffer, base, y0, x0 + 1, width, height, mode, border_value), fx),
		lerp(image_sample_bordered(buffer, base, y0 + 1, x0, width, height, mode, border_value),
			image_sample_bordered(buffer, base, y0 + 1, x0 + 1, width, height, mode, border_value), fx),
		fy);
}

// The pad donor's fetch: output coordinates offset by the pad amounts
// and folded through the border mode before the plane read.
[ForceInline]
export float image_pad_fetch(
	RWByteAddressBuffer buffer,
	uint base,
	uint x,
	uint y,
	uint pad_left,
	uint pad_top,
	uint width,
	uint height,
	uint mode,
	float border_value
) {
	int source_x = image_border_coordinate(int(x) - int(pad_left), width, mode);
	int source_y = image_border_coordinate(int(y) - int(pad_top), height, mode);
	if (source_x < 0 || source_y < 0) return border_value;
	return load_f32(buffer, base + uint(source_y) * width + uint(source_x));
}

// The warp donors' fetch: the inverse transform row loads, the source
// coordinate (affine 2x3 or perspective-divided 3x3), then the bordered
// gather. `perspective != 0` selects the 3x3 path with the donor's
// denominator guard.
[ForceInline]
export float image_warp_f32(
	RWByteAddressBuffer input,
	RWByteAddressBuffer transform,
	uint base,
	uint x,
	uint y,
	uint width,
	uint height,
	uint interp,
	uint mode,
	float border_value,
	uint perspective
) {
	float m0 = load_f32(transform, 0u);
	float m1 = load_f32(transform, 1u);
	float m2 = load_f32(transform, 2u);
	float m3 = load_f32(transform, 3u);
	float m4 = load_f32(transform, 4u);
	float m5 = load_f32(transform, 5u);
	float source_x = m0 * float(x) + m1 * float(y) + m2;
	float source_y = m3 * float(x) + m4 * float(y) + m5;
	if (perspective != 0u) {
		float m6 = load_f32(transform, 6u);
		float m7 = load_f32(transform, 7u);
		float m8 = load_f32(transform, 8u);
		float denominator = m6 * float(x) + m7 * float(y) + m8;
		source_x = -1.0e20f;
		source_y = -1.0e20f;
		if (abs(denominator) >= 1.0e-20f) {
			source_x = (m0 * float(x) + m1 * float(y) + m2) / denominator;
			source_y = (m3 * float(x) + m4 * float(y) + m5) / denominator;
		}
	}
	return image_gather_bordered(
		input, base, source_y, source_x, width, height, interp, mode, border_value);
}

// The remap donor's fetch: absolute (x, y) source coordinates read from
// the two map planes, then the bordered gather.
[ForceInline]
export float image_remap_f32(
	RWByteAddressBuffer input,
	RWByteAddressBuffer map,
	uint base,
	uint x,
	uint y,
	uint batch,
	uint width,
	uint height,
	uint out_width,
	uint out_height,
	uint interp,
	uint mode,
	uint map_batch,
	float border_value
) {
	uint map_plane = out_height * out_width;
	uint map_base = (map_batch == 1u ? 0u : batch) * 2u * map_plane + y * out_width + x;
	float source_x = load_f32(map, map_base);
	float source_y = load_f32(map, map_base + map_plane);
	return image_gather_bordered(
		input, base, source_y, source_x, width, height, interp, mode, border_value);
}
