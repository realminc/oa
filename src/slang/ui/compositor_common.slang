// Shared UI compositor color ABI and premultiplied-alpha operation.
//
// This stays in the UI domain: RGBA push packing and compositor blending are
// not generic numerical math and must not leak into DNN kernel modules.

float4 uiUnpackRgba(uint rgba) {
	return float4(
		float((rgba >> 24) & 0xffu) / 255.0,
		float((rgba >> 16) & 0xffu) / 255.0,
		float((rgba >>  8) & 0xffu) / 255.0,
		float(rgba         & 0xffu) / 255.0);
}

float4 uiSourceOver(float4 src, float4 dst) {
	return float4(
		src.rgb * src.a + dst.rgb * (1.0 - src.a),
		src.a + dst.a * (1.0 - src.a));
}
