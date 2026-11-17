// Shared sampling and color math for the to_matrix kernel family.
//
// Source coordinates are continuous luma-pixel-center positions. Chroma
// planes are sampled with centered 4:2:0 siting: chroma sample i covers
// luma columns 2i and 2i+1, so its center sits at luma coordinate
// 2i + 0.5 and the chroma coordinate of luma position s is (s - 0.5) * 0.5.

import storage;
import yuv_to_rgb;

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

// Bilinear (interp == 0) or nearest (interp == 1) sample of one u8 plane at
// continuous pixel-center coordinates, clamped to the plane extent.
export float to_matrix_sample_plane(
	RWByteAddressBuffer plane,
	float x,
	float y,
	uint plane_width,
	uint plane_height,
	uint interp
) {
	if (interp != 0u) {
		int xi = clamp(int(floor(x + 0.5f)), 0, int(plane_width) - 1);
		int yi = clamp(int(floor(y + 0.5f)), 0, int(plane_height) - 1);
		return float(load_u8(plane, uint(yi) * plane_width + uint(xi))) / 255.0f;
	}
	float xc = clamp(x, 0.0f, float(plane_width) - 1.0f);
	float yc = clamp(y, 0.0f, float(plane_height) - 1.0f);
	uint x0 = min(uint(floor(xc)), plane_width - 1u);
	uint y0 = min(uint(floor(yc)), plane_height - 1u);
	uint x1 = min(x0 + 1u, plane_width - 1u);
	uint y1 = min(y0 + 1u, plane_height - 1u);
	float fx = xc - float(x0);
	float fy = yc - float(y0);
	float top_left = float(load_u8(plane, y0 * plane_width + x0)) / 255.0f;
	float top_right = float(load_u8(plane, y0 * plane_width + x1)) / 255.0f;
	float bottom_left = float(load_u8(plane, y1 * plane_width + x0)) / 255.0f;
	float bottom_right = float(load_u8(plane, y1 * plane_width + x1)) / 255.0f;
	float top = lerp(top_left, top_right, fx);
	float bottom = lerp(bottom_left, bottom_right, fx);
	return lerp(top, bottom, fy);
}

// Sample one component of an interleaved NV12 chroma plane. `component` is
// 0 for Cb and 1 for Cr; the UV row is 2 * chroma_width bytes wide.
export float to_matrix_sample_nv12_chroma(
	RWByteAddressBuffer uv_plane,
	float x,
	float y,
	uint chroma_width,
	uint chroma_height,
	uint component,
	uint interp
) {
	if (interp != 0u) {
		int xi = clamp(int(floor(x + 0.5f)), 0, int(chroma_width) - 1);
		int yi = clamp(int(floor(y + 0.5f)), 0, int(chroma_height) - 1);
		return float(
			load_u8(uv_plane, (uint(yi) * chroma_width + uint(xi)) * 2u + component)
		) / 255.0f;
	}
	float xc = clamp(x, 0.0f, float(chroma_width) - 1.0f);
	float yc = clamp(y, 0.0f, float(chroma_height) - 1.0f);
	uint x0 = min(uint(floor(xc)), chroma_width - 1u);
	uint y0 = min(uint(floor(yc)), chroma_height - 1u);
	uint x1 = min(x0 + 1u, chroma_width - 1u);
	uint y1 = min(y0 + 1u, chroma_height - 1u);
	float fx = xc - float(x0);
	float fy = yc - float(y0);
	float top_left = float(load_u8(uv_plane, (y0 * chroma_width + x0) * 2u + component)) / 255.0f;
	float top_right = float(load_u8(uv_plane, (y0 * chroma_width + x1) * 2u + component)) / 255.0f;
	float bottom_left = float(load_u8(uv_plane, (y1 * chroma_width + x0) * 2u + component)) / 255.0f;
	float bottom_right = float(load_u8(uv_plane, (y1 * chroma_width + x1) * 2u + component)) / 255.0f;
	float top = lerp(top_left, top_right, fx);
	float bottom = lerp(bottom_left, bottom_right, fx);
	return lerp(top, bottom, fy);
}
