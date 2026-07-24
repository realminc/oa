// SDF glyph sampling from the bindless atlas buffer.

struct FragmentInput {
	float2 atlas_position : TEXCOORD0;
	float4 color : COLOR0;
	float smoothing : TEXCOORD1;
	nointerpolation uint solid : TEXCOORD2;
};

struct PushConstants {
	float4x4 mvp;
	uint glyph_idx;
	uint atlas_idx;
	uint reference_width;
	uint reference_height;
	uint atlas_width;
	uint atlas_height;
	float px_range;
	uint reserved0;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] ByteAddressBuffer heap[];

float loadAtlas(int x, int y) {
	x = clamp(x, 0, int(pc.atlas_width) - 1);
	y = clamp(y, 0, int(pc.atlas_height) - 1);
	uint byteOffset = uint(y) * pc.atlas_width + uint(x);
	uint word = heap[pc.atlas_idx].Load(byteOffset & ~3u);
	return float((word >> ((byteOffset & 3u) * 8u)) & 0xffu) / 255.0;
}

float sampleAtlas(float2 position) {
	int2 p0 = int2(floor(position));
	float2 f = frac(position);
	float a = lerp(loadAtlas(p0.x, p0.y), loadAtlas(p0.x + 1, p0.y), f.x);
	float b = lerp(loadAtlas(p0.x, p0.y + 1), loadAtlas(p0.x + 1, p0.y + 1), f.x);
	return lerp(a, b, f.y);
}

[shader("fragment")]
float4 main(FragmentInput input) : SV_Target0 {
	if (input.solid != 0) return input.color;
	float distance = sampleAtlas(input.atlas_position);
	float alpha = smoothstep(
		0.5 - input.smoothing,
		0.5 + input.smoothing,
		distance) * input.color.a;
	return float4(input.color.rgb, alpha);
}
