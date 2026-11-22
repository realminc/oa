// UI raster RGBA8 image blit — sampled path.
//
// Replaces the packed-buffer blit_rgba.frag: the source Texture's native
// R8G8B8A8_UNORM image is sampled through a combined image sampler bound per
// command. The scissor covers clip ∩ dst; the fragment bounds-checks the
// destination-local index as a second guard.
//
// Sampling rule (host picks the bound sampler, `filter` selects the math):
//   filter 0 — exact nearest: the donor's integer x * src / dst texel index,
//              sampled at the texel centre through a NEAREST sampler. This
//              reproduces the packed-buffer path bit for bit.
//   filter 1 — bilinear: uv at the destination pixel centre through a LINEAR
//              sampler; used when the destination downscales the source.
//
// Push layout (32 bytes):
//   offset  0 : uint2 src_size
//   offset  8 : int2  dst_origin
//   offset 16 : uint2 dst_size
//   offset 24 : uint  channel — 0 = RGBA; 1..4 select one channel to RGB
//   offset 28 : uint  filter  — 0 = nearest, 1 = bilinear

struct PushConstants {
	uint src_w;
	uint src_h;
	int dst_x;
	int dst_y;
	uint dst_w;
	uint dst_h;
	uint channel;
	uint filter;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] Sampler2D source;

[shader("fragment")]
float4 main(float4 position : SV_Position) : SV_Target0 {
	uint x = uint(int(position.x) - pc.dst_x);
	uint y = uint(int(position.y) - pc.dst_y);
	if (x >= pc.dst_w || y >= pc.dst_h || pc.src_w == 0 || pc.src_h == 0) {
		discard;
	}

	float2 uv;
	if (pc.filter == 0u) {
		uint sx = min(x * pc.src_w / pc.dst_w, pc.src_w - 1);
		uint sy = min(y * pc.src_h / pc.dst_h, pc.src_h - 1);
		uv = (float2(float(sx), float(sy)) + 0.5)
			/ float2(float(pc.src_w), float(pc.src_h));
	} else {
		uv = (float2(float(x), float(y)) + 0.5)
			/ float2(float(pc.dst_w), float(pc.dst_h));
	}

	float4 src = source.Sample(uv);
	if (pc.channel != 0u) {
		float selected = pc.channel == 1u
			? src.r : pc.channel == 2u ? src.g : pc.channel == 3u ? src.b : src.a;
		src = float4(selected, selected, selected, 1.0);
	}
	return float4(src.rgb * src.a, src.a);
}
