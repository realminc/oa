// Camera-anchored, pixel-sized glyph and label background quads.

struct VertexOutput {
	float4 sv_position : SV_Position;
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

float4 unpackRgba(uint rgba) {
	return float4(
		float((rgba >> 24) & 0xff) / 255.0,
		float((rgba >> 16) & 0xff) / 255.0,
		float((rgba >> 8) & 0xff) / 255.0,
		float(rgba & 0xff) / 255.0);
}

[shader("vertex")]
VertexOutput main(uint vertexId : SV_VertexID, uint instanceId : SV_InstanceID) {
	uint offset = instanceId * 48;
	float anchorX = asfloat(heap[pc.glyph_idx].Load(offset));
	float anchorY = asfloat(heap[pc.glyph_idx].Load(offset + 4));
	float offsetX = asfloat(heap[pc.glyph_idx].Load(offset + 8));
	float offsetY = asfloat(heap[pc.glyph_idx].Load(offset + 12));
	float width = asfloat(heap[pc.glyph_idx].Load(offset + 16));
	float height = asfloat(heap[pc.glyph_idx].Load(offset + 20));
	uint atlasX = heap[pc.glyph_idx].Load(offset + 24);
	uint atlasY = heap[pc.glyph_idx].Load(offset + 28);
	uint atlasW = heap[pc.glyph_idx].Load(offset + 32);
	uint atlasH = heap[pc.glyph_idx].Load(offset + 36);
	float4 color = unpackRgba(heap[pc.glyph_idx].Load(offset + 40));

	static const float2 corners[6] = {
		float2(0.0, 0.0), float2(1.0, 0.0), float2(1.0, 1.0),
		float2(1.0, 1.0), float2(0.0, 1.0), float2(0.0, 0.0),
	};
	float2 corner = corners[vertexId];
	float2 pixelOffset = float2(offsetX, offsetY) + corner * float2(width, height);
	float2 sourcePosition = float2(anchorX, anchorY) + float2(
		pixelOffset.x / float(pc.reference_width),
		pixelOffset.y / float(pc.reference_height));
	float2 worldPosition = float2(
		sourcePosition.x - 0.5,
		0.5 - sourcePosition.y);
	float4 clip = mul(float4(worldPosition, 0.0, 1.0), pc.mvp);

	VertexOutput output;
	output.sv_position = clip;
	output.atlas_position =
		float2(float(atlasX), float(atlasY))
		+ corner * float2(float(atlasW), float(atlasH)) - 0.5;
	output.color = color;
	output.smoothing = atlasW == 0 || atlasH == 0
		? 0.0
		: clamp(
			0.5 * max(float(atlasW) / width, float(atlasH) / height)
				/ pc.px_range,
			0.01, 0.25);
	output.solid = atlasW == 0 || atlasH == 0;
	return output;
}
