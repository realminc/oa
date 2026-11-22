// UI raster glyph instances — vertex stage.
//
// One instanced quad per 48-byte coverage/SDF glyph record, replacing the
// draw_glyphs.slang per-glyph workgroup. The vertex stage reproduces the
// compute anchor/quad math, culls non-matching batches to a degenerate
// triangle, and forwards the atlas rectangle and smoothing for the fragment.
//
// Push layout (44 bytes):
//   offset  0 : int2  dst_origin        normalized anchor target rect
//   offset  8 : uint2 dst_size
//   offset 16 : uint2 image_size        compose extent (pixels)
//   offset 24 : uint  first             first record index
//   offset 28 : uint  batch             admitted (flags >> 1) value
//   offset 32 : uint2 atlas_size        coverage atlas extent (pixels)
//   offset 40 : float px_range          SDF pixel range; 0 = coverage atlas

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

[shader("vertex")]
GlyphVaryings main(
	uint vertex_id : SV_VertexID,
	uint instance_id : SV_InstanceID)
{
	GlyphVaryings output;
	uint offset = (pc.first + instance_id) * 48u;
	float anchor_x = asfloat(heap[0].Load(offset));
	float anchor_y = asfloat(heap[0].Load(offset + 4));
	float offset_x = asfloat(heap[0].Load(offset + 8));
	float offset_y = asfloat(heap[0].Load(offset + 12));
	float width = asfloat(heap[0].Load(offset + 16));
	float height = asfloat(heap[0].Load(offset + 20));
	uint atlas_x = heap[0].Load(offset + 24);
	uint atlas_y = heap[0].Load(offset + 28);
	uint atlas_w = heap[0].Load(offset + 32);
	uint atlas_h = heap[0].Load(offset + 36);
	uint packed_color = heap[0].Load(offset + 40);
	uint flags = heap[0].Load(offset + 44);

	bool bottom_to_top = (flags & 1u) != 0u;
	int x0 = int(floor(float(pc.dst_x) + anchor_x * float(pc.dst_w) + offset_x));
	int y0 = int(floor(float(pc.dst_y) + anchor_y * float(pc.dst_h) + offset_y));
	uint quad_w = uint(max(1.0, ceil(bottom_to_top ? height : width)));
	uint quad_h = uint(max(1.0, ceil(bottom_to_top ? width : height)));

	float smoothing = 0.0;
	bool solid = atlas_w == 0u || atlas_h == 0u;
	if (!solid && pc.px_range > 0.0) {
		smoothing = clamp(
			0.5 * max(float(atlas_w) / width, float(atlas_h) / height)
				/ pc.px_range,
			0.01, 0.25);
	}

	const float2 corners[6] = {
		float2(0.0, 0.0), float2(1.0, 0.0), float2(0.0, 1.0),
		float2(0.0, 1.0), float2(1.0, 0.0), float2(1.0, 1.0),
	};
	float2 corner = corners[vertex_id % 6u];
	// Batch filter: collapse the quad when this record belongs to another batch.
	if ((flags >> 1) != pc.batch) {
		corner = float2(0.0, 0.0);
		quad_w = 0u;
		quad_h = 0u;
	}
	float2 pixel = float2(float(x0), float(y0))
		+ corner * float2(float(quad_w), float(quad_h));
	float2 ndc = float2(
		pixel.x / float(pc.image_w) * 2.0 - 1.0,
		pixel.y / float(pc.image_h) * 2.0 - 1.0);

	output.position = float4(ndc, 0.0, 1.0);
	output.quad_origin = float2(float(x0), float(y0));
	output.quad_size = float2(float(quad_w), float(quad_h));
	output.atlas_rect = float4(
		float(atlas_x), float(atlas_y), float(atlas_w), float(atlas_h));
	output.color = float4(
		float((packed_color >> 24) & 0xffu) / 255.0,
		float((packed_color >> 16) & 0xffu) / 255.0,
		float((packed_color >> 8) & 0xffu) / 255.0,
		float(packed_color & 0xffu) / 255.0);
	output.record_size = float2(width, height);
	output.flags = flags;
	output.smoothing = smoothing;
	return output;
}
