// UI raster glyph instances — fragment stage.
//
// Per-pixel counterpart of the draw_glyphs.slang inner loop: reconstructs the
// quad-local coordinate from SV_Position, samples the coverage/SDF atlas at
// the same bilinear position, and returns premultiplied alpha for the fixed
// source-over blend.

struct PushConstants {
	int dst_x;
	int dst_y;
	uint dst_w;
	uint dst_h;
	uint image_w;
	uint image_h;
	uint first;
	uint batch;
	uint atlas_w;
	uint atlas_h;
	float px_range;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] ByteAddressBuffer heap[2];

struct GlyphVaryings {
	float4 position : SV_Position;
	float2 quad_origin : QUAD_ORIGIN;
	float2 quad_size : QUAD_SIZE;
	nointerpolation float4 atlas_rect : ATLAS_RECT;
	nointerpolation float4 color : GLYPH_COLOR;
	nointerpolation float2 record_size : RECORD_SIZE;
	nointerpolation uint flags : GLYPH_FLAGS;
	nointerpolation float smoothing : SMOOTHING;
};

float load_atlas(int x, int y) {
	x = clamp(x, 0, int(pc.atlas_w) - 1);
	y = clamp(y, 0, int(pc.atlas_h) - 1);
	uint offset = uint(y) * pc.atlas_w + uint(x);
	uint word = heap[1].Load(offset & ~3u);
	return float((word >> ((offset & 3u) * 8u)) & 255u) / 255.0;
}

float sample_atlas(float2 position) {
	int2 p = int2(floor(position));
	float2 f = frac(position);
	float a = lerp(load_atlas(p.x, p.y), load_atlas(p.x + 1, p.y), f.x);
	float b = lerp(load_atlas(p.x, p.y + 1), load_atlas(p.x + 1, p.y + 1), f.x);
	return lerp(a, b, f.y);
}

[shader("fragment")]
float4 main(GlyphVaryings input) : SV_Target0 {
	float alpha = input.color.a;
	bool bottom_to_top = (input.flags & 1u) != 0u;
	bool solid = input.atlas_rect.z == 0.0 || input.atlas_rect.w == 0.0;
	if (!solid) {
		float2 local = input.position.xy - input.quad_origin;
		float width = input.record_size.x;
		float height = input.record_size.y;
		float2 source_position = bottom_to_top
			? float2(width - local.y, local.x)
			: local;
		float2 uv = source_position / float2(width, height);
		float2 atlas_position = input.atlas_rect.xy
			+ uv * input.atlas_rect.zw - 0.5;
		float coverage = sample_atlas(atlas_position);
		alpha *= pc.px_range > 0.0
			? smoothstep(0.5 - input.smoothing, 0.5 + input.smoothing, coverage)
			: coverage;
	}
	if (alpha <= 0.001) discard;
	return float4(input.color.rgb * alpha, alpha);
}
