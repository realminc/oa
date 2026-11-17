// UI raster audio waveform — precomputed [bins, 2] min/max envelope columns.
//
// Raster port of draw_waveform.slang. The scissor covers the element rect
// intersected with its clip; each pixel has at most one source contribution,
// so premultiplied output through the fixed blend is bit-exact.

import storage;

struct PushConstants {
	uint bins;
	int dst_x;
	int dst_y;
	uint dst_w;
	uint dst_h;
	float fraction;
	uint played_rgba;
	uint remaining_rgba;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] ByteAddressBuffer heap[1];

float4 unpack_rgba(uint rgba) {
	return float4(
		float((rgba >> 24) & 0xff) / 255.0,
		float((rgba >> 16) & 0xff) / 255.0,
		float((rgba >> 8) & 0xff) / 255.0,
		float(rgba & 0xff) / 255.0);
}

[shader("fragment")]
float4 main(float4 position : SV_Position) : SV_Target0 {
	uint x = uint(int(position.x) - pc.dst_x);
	int y = int(position.y) - pc.dst_y;
	if (x >= pc.dst_w || pc.bins == 0 || pc.dst_h == 0) discard;

	uint bin = min(pc.bins - 1, x * pc.bins / pc.dst_w);
	float minimum = min(0.0, clamp(oaLoad(heap[0], bin * 2), -1.0, 1.0));
	float maximum = max(0.0, clamp(oaLoad(heap[0], bin * 2 + 1), -1.0, 1.0));

	float half_height = float(pc.dst_h - 1) * 0.5;
	int y0 = int(round(half_height - maximum * half_height));
	int y1 = int(round(half_height - minimum * half_height));
	y0 = clamp(y0, 0, int(pc.dst_h) - 1);
	y1 = clamp(y1, y0, int(pc.dst_h) - 1);
	if (y < y0 || y > y1) discard;

	float progress = (float(x) + 0.5) / float(pc.dst_w);
	float4 color = unpack_rgba(progress <= pc.fraction
		? pc.played_rgba
		: pc.remaining_rgba);
	return float4(color.rgb * color.a, color.a);
}
