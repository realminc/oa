// Shared Y'CbCr -> R'G'B' conversion math for the to_matrix kernel family
// and the presentation shader fallback.
//
// Inverse of rgb_to_yuv.slang. Inputs are range-normalized components:
// luma in [0,1] and chroma centered on zero in [-0.5,0.5]. Range expansion
// happens in to_matrix_common.slang before conversion, so one formula set
// serves limited- and full-range sources.

export static const uint COLOR_SPACE_BT601 = 0;
export static const uint COLOR_SPACE_BT709 = 1;
export static const uint COLOR_SPACE_BT2020 = 2;

export float3 yuv_to_rgb_bt601(float3 yuv) {
	float r = yuv.x + 1.402f * yuv.z;
	float g = yuv.x - 0.344136f * yuv.y - 0.714136f * yuv.z;
	float b = yuv.x + 1.772f * yuv.y;
	return saturate(float3(r, g, b));
}

export float3 yuv_to_rgb_bt709(float3 yuv) {
	float r = yuv.x + 1.5748f * yuv.z;
	float g = yuv.x - 0.187324f * yuv.y - 0.468124f * yuv.z;
	float b = yuv.x + 1.8556f * yuv.y;
	return saturate(float3(r, g, b));
}

export float3 yuv_to_rgb_bt2020(float3 yuv) {
	float r = yuv.x + 1.4746f * yuv.z;
	float g = yuv.x - 0.164553f * yuv.y - 0.571353f * yuv.z;
	float b = yuv.x + 1.8814f * yuv.y;
	return saturate(float3(r, g, b));
}

export float3 yuv_to_rgb(float3 yuv, uint color_space) {
	if (color_space == COLOR_SPACE_BT709) {
		return yuv_to_rgb_bt709(yuv);
	}
	if (color_space == COLOR_SPACE_BT2020) {
		return yuv_to_rgb_bt2020(yuv);
	}
	return yuv_to_rgb_bt601(yuv);
}
