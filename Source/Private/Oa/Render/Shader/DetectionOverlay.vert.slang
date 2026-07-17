// Camera-bound detection rectangle borders.

struct VertexOutput {
	float4 sv_position : SV_Position;
	float4 color       : COLOR0;
};

struct PushConstants {
	float4x4 mvp;
	float4 color;
	uint detection_idx;
	uint target_width;
	uint target_height;
	float thickness_pixels;
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
	uint offset = instanceId * 32;
	float centerX = asfloat(heap[pc.detection_idx].Load(offset));
	float centerY = asfloat(heap[pc.detection_idx].Load(offset + 4));
	float width = asfloat(heap[pc.detection_idx].Load(offset + 8));
	float height = asfloat(heap[pc.detection_idx].Load(offset + 12));
	uint instanceRgba = heap[pc.detection_idx].Load(offset + 24);

	float left = centerX - width * 0.5;
	float right = centerX + width * 0.5;
	float top = 0.5 - centerY + height * 0.5;
	float bottom = 0.5 - centerY - height * 0.5;
	left -= 0.5;
	right -= 0.5;

	uint edge = vertexId / 6;
	uint corner = vertexId % 6;
	float2 edgeStart;
	float2 edgeEnd;
	if (edge == 0) {
		edgeStart = float2(left, top);
		edgeEnd = float2(right, top);
	} else if (edge == 1) {
		edgeStart = float2(right, top);
		edgeEnd = float2(right, bottom);
	} else if (edge == 2) {
		edgeStart = float2(right, bottom);
		edgeEnd = float2(left, bottom);
	} else {
		edgeStart = float2(left, bottom);
		edgeEnd = float2(left, top);
	}

	float4 clipStart = mul(float4(edgeStart, 0.0, 1.0), pc.mvp);
	float4 clipEnd = mul(float4(edgeEnd, 0.0, 1.0), pc.mvp);
	float2 startNdc = clipStart.xy / clipStart.w;
	float2 endNdc = clipEnd.xy / clipEnd.w;
	float2 delta = endNdc - startNdc;
	float2 direction = delta * rsqrt(max(dot(delta, delta), 1.0e-12));
	float2 normal = float2(-direction.y, direction.x);
	float2 ndcPerPixel = float2(
		2.0 / float(pc.target_width),
		2.0 / float(pc.target_height));
	float2 offsetNdc = normal * ndcPerPixel * pc.thickness_pixels * 0.5;

	static const float2 corners[6] = {
		float2(0.0, -1.0), float2(1.0, -1.0), float2(1.0, 1.0),
		float2(1.0, 1.0), float2(0.0, 1.0), float2(0.0, -1.0),
	};
	float2 cornerData = corners[corner];
	float4 clip = cornerData.x < 0.5 ? clipStart : clipEnd;
	clip.xy += offsetNdc * cornerData.y * clip.w;

	VertexOutput output;
	output.sv_position = clip;
	output.color = instanceRgba != 0 ? unpackRgba(instanceRgba) : pc.color;
	return output;
}
