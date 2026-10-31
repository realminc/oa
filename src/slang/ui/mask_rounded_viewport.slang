// Donor: oacpp/source/cpp/lib/oa/ui/shader/compute/maskRoundedViewport.slang
// Adaptation: spelling/indentation and explicit RGBA8 storage format only.
// MaskRoundedViewport.slang — premultiplied-alpha rounded mask for a local
// region or the complete client-side window composition.

struct PushConstants {
	uint dst_idx;
	int dst_x;
	int dst_y;
	uint dst_w;
	uint dst_h;
	float corner_radius;
	uint outside_rgba;
	int clip_x;
	int clip_y;
	uint clip_w;
	uint clip_h;
};

[[vk::push_constant]] PushConstants pc;
[[vk::image_format("rgba8")]]
[[vk::binding(1, 0)]] RWTexture2D<float4> images[];

float rounded_viewport_coverage(uint x, uint y) {
	float radius = min(
		max(pc.corner_radius, 0.0),
		0.5 * float(min(pc.dst_w, pc.dst_h)));
	if (radius <= 0.0) return 1.0;
	float2 half_size = 0.5 * float2(pc.dst_w, pc.dst_h);
	float2 local = float2(x, y) + 0.5 - half_size;
	float2 q = abs(local) - (half_size - radius);
	float distance = length(max(q, float2(0.0, 0.0)))
		+ min(max(q.x, q.y), 0.0) - radius;
	return saturate(0.5 - distance);
}

float4 unpack_premultiplied_rgba(uint rgba) {
	float4 color = float4(
		float((rgba >> 24) & 0xff) / 255.0,
		float((rgba >> 16) & 0xff) / 255.0,
		float((rgba >> 8) & 0xff) / 255.0,
		float(rgba & 0xff) / 255.0);
	return float4(color.rgb * color.a, color.a);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
	uint x = gid.x * 8 + lid.x;
	uint y = gid.y * 8 + lid.y;
	if (x >= pc.dst_w || y >= pc.dst_h) return;

	int dst_x = pc.dst_x + int(x);
	int dst_y = pc.dst_y + int(y);
	if (dst_x < pc.clip_x || dst_y < pc.clip_y
		|| uint(dst_x - pc.clip_x) >= pc.clip_w
		|| uint(dst_y - pc.clip_y) >= pc.clip_h) return;
	uint image_width, image_height;
	images[pc.dst_idx].GetDimensions(image_width, image_height);
	if (dst_x < 0 || dst_y < 0
		|| dst_x >= int(image_width) || dst_y >= int(image_height)) return;

	float coverage = rounded_viewport_coverage(x, y);
	uint2 dst_coord = uint2(uint(dst_x), uint(dst_y));
	float4 current = images[pc.dst_idx][dst_coord];
	float4 outside = unpack_premultiplied_rgba(pc.outside_rgba);
	images[pc.dst_idx][dst_coord] =
		current * coverage + outside * (1.0 - coverage);
}
