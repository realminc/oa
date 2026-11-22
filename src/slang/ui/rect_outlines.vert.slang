// UI raster batched normalized rectangle borders — vertex stage.
//
// One instanced quad per 32-byte detection record, replacing the
// draw_rect_outlines.slang per-record workgroup. The vertex stage reproduces
// the compute border-rect math and forwards the clamped border width and
// resolved color; the fragment writes the border pixels the compute path
// stored — the pipeline is replace (blend-disabled) to match.
//
// Push layout (32 bytes):
//   offset  0 : int2  dst_origin        normalized record target rect
//   offset  8 : uint2 dst_size
//   offset 16 : uint2 image_size        compose extent (pixels)
//   offset 24 : uint  thickness         requested border thickness (pixels)
//   offset 28 : uint  rgba              fallback color when the record's is 0

struct PushConstants {
	int dst_x;
	int dst_y;
	uint dst_w;
	uint dst_h;
	uint image_w;
	uint image_h;
	uint thickness;
	uint rgba;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] ByteAddressBuffer heap[1];

struct OutlineVaryings {
	float4 position : SV_Position;
	nointerpolation float4 rect : OUTLINE_RECT;
	nointerpolation float4 color : OUTLINE_COLOR;
	nointerpolation uint border : OUTLINE_BORDER;
};

[shader("vertex")]
OutlineVaryings main(
	uint vertex_id : SV_VertexID,
	uint instance_id : SV_InstanceID)
{
	OutlineVaryings output;
	uint offset = instance_id * 32u;
	float center_x = asfloat(heap[0].Load(offset));
	float center_y = asfloat(heap[0].Load(offset + 4));
	float width = asfloat(heap[0].Load(offset + 8));
	float height = asfloat(heap[0].Load(offset + 12));
	uint instance_rgba = heap[0].Load(offset + 24);

	int x0 = int(floor(float(pc.dst_x) + (center_x - width * 0.5) * float(pc.dst_w)));
	int y0 = int(floor(float(pc.dst_y) + (center_y - height * 0.5) * float(pc.dst_h)));
	int x1 = int(ceil(float(pc.dst_x) + (center_x + width * 0.5) * float(pc.dst_w)));
	int y1 = int(ceil(float(pc.dst_y) + (center_y + height * 0.5) * float(pc.dst_h)));

	uint packed_color = instance_rgba != 0u ? instance_rgba : pc.rgba;
	int rect_w = x1 - x0;
	int rect_h = y1 - y0;
	uint border = min(pc.thickness, uint((min(rect_w, rect_h) + 1) / 2));

	const float2 corners[6] = {
		float2(0.0, 0.0), float2(1.0, 0.0), float2(0.0, 1.0),
		float2(0.0, 1.0), float2(1.0, 0.0), float2(1.0, 1.0),
	};
	float2 corner = rect_w <= 0 || rect_h <= 0
		? float2(0.0, 0.0)
		: corners[vertex_id % 6u];
	float2 pixel = float2(float(x0), float(y0))
		+ corner * float2(float(x1 - x0), float(y1 - y0));
	float2 ndc = float2(
		pixel.x / float(pc.image_w) * 2.0 - 1.0,
		pixel.y / float(pc.image_h) * 2.0 - 1.0);

	output.position = float4(ndc, 0.0, 1.0);
	output.rect = float4(float(x0), float(y0), float(x1), float(y1));
	output.color = float4(
		float((packed_color >> 24) & 0xffu) / 255.0,
		float((packed_color >> 16) & 0xffu) / 255.0,
		float((packed_color >> 8) & 0xffu) / 255.0,
		float(packed_color & 0xffu) / 255.0);
	output.border = border;
	return output;
}
