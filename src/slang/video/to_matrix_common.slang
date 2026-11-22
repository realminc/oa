// Shared sampling and color math for the to_matrix kernel family.
//
// Source coordinates are continuous luma-pixel-center positions. Chroma
// planes are sampled with centered 4:2:0 siting: chroma sample i covers
// luma columns 2i and 2i+1, so its center sits at luma coordinate
// 2i + 0.5 and the chroma coordinate of luma position s is (s - 0.5) * 0.5.

import storage;

// Scalar channel modes matching VideoMatrixConfig::channels.
static const uint CHANNELS_LUMA = 0;
static const uint CHANNELS_RGB = 1;
static const uint CHANNELS_BGR = 2;
static const uint CHANNELS_YUV = 3;

// Packed color_mode values matching the host encoding: color_space * 2 +
// full_range.
static const uint COLOR_MODE_FULL_RANGE = 1;

// Expand one normalized code value to [0,1] luma under the frame's range.
export float to_matrix_scale_luma(float code, uint color_mode) {
	if ((color_mode & COLOR_MODE_FULL_RANGE) != 0u) {
		return code;
	}
	return clamp((code * 255.0f - 16.0f) / 219.0f, 0.0f, 1.0f);
}

// Expand one normalized code value to zero-centered chroma under the range.
export float to_matrix_scale_chroma(float code, uint color_mode) {
	if ((color_mode & COLOR_MODE_FULL_RANGE) != 0u) {
		return code - 0.5f;
	}
	return clamp((code * 255.0f - 128.0f) / 224.0f, -0.5f, 0.5f);
}

// NV12 stores byte codes. P010 stores a little-endian 16-bit lane with
// ten significant bits at the high end. Read an aligned word so the last
// component of an odd-sized plane uses only the allocation's padding.
[ForceInline]
float to_matrix_code(RWByteAddressBuffer plane, uint element, uint sample_bits) {
	if (sample_bits == 8u) {
		return float(load_u8(plane, element)) / 255.0f;
	}
	uint word = plane.Load((element & ~1u) * 2u);
	uint code = (word >> ((element & 1u) * 16u + 6u)) & 1023u;
	return float(code) / 1023.0f;
}

// Bilinear (interp == 0) or nearest sampling of one component, with
// coordinates clamped to the component plane's pixel centers.
[ForceInline]
export float to_matrix_sample_component(
	RWByteAddressBuffer plane,
	float x,
	float y,
	uint width,
	uint height,
	uint components,
	uint component,
	uint interp,
	uint sample_bits
) {
	if (interp != 0u) {
		int xi = clamp(int(floor(x + 0.5f)), 0, int(width) - 1);
		int yi = clamp(int(floor(y + 0.5f)), 0, int(height) - 1);
		uint index = (uint(yi) * width + uint(xi)) * components + component;
		return to_matrix_code(plane, index, sample_bits);
	}
	float xc = clamp(x, 0.0f, float(width) - 1.0f);
	float yc = clamp(y, 0.0f, float(height) - 1.0f);
	uint x0 = min(uint(floor(xc)), width - 1u);
	uint y0 = min(uint(floor(yc)), height - 1u);
	uint x1 = min(x0 + 1u, width - 1u);
	uint y1 = min(y0 + 1u, height - 1u);
	float fx = xc - float(x0);
	float fy = yc - float(y0);
	uint row0 = y0 * width;
	uint row1 = y1 * width;
	float top_left = to_matrix_code(plane, (row0 + x0) * components + component, sample_bits);
	float top_right = to_matrix_code(plane, (row0 + x1) * components + component, sample_bits);
	float bottom_left = to_matrix_code(plane, (row1 + x0) * components + component, sample_bits);
	float bottom_right = to_matrix_code(plane, (row1 + x1) * components + component, sample_bits);
	float top = lerp(top_left, top_right, fx);
	float bottom = lerp(bottom_left, bottom_right, fx);
	return lerp(top, bottom, fy);
}

export float to_matrix_sample_plane(
	RWByteAddressBuffer plane,
	float x,
	float y,
	uint width,
	uint height,
	uint interp
) {
	return to_matrix_sample_component(plane, x, y, width, height, 1u, 0u, interp, 8u);
}

export float to_matrix_scale_p010_luma(float code, uint color_mode) {
	if ((color_mode & COLOR_MODE_FULL_RANGE) != 0u) {
		return code;
	}
	return clamp((code * 1023.0f - 64.0f) / 876.0f, 0.0f, 1.0f);
}

export float to_matrix_scale_p010_chroma(float code, uint color_mode) {
	if ((color_mode & COLOR_MODE_FULL_RANGE) != 0u) {
		return clamp(code - 512.0f / 1023.0f, -0.5f, 0.5f);
	}
	return clamp((code * 1023.0f - 512.0f) / 896.0f, -0.5f, 0.5f);
}
