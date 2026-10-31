// Donor: oacpp/source/cpp/lib/oa/ui/shader/compute/drawHeatmap.slang
// Adaptation: naming/formatting and explicit RGBA8 storage image format.
// Dense matrix heatmap rendered directly from a bindless GPU buffer.

struct PushConstants {
	uint values_idx;
	uint rows;
	uint cols;
	uint dst_idx;
	int dst_x;
	int dst_y;
	uint dst_w;
	uint dst_h;
	float v_min;
	float v_max;
	uint colormap;
	uint value_type; // 0=fp32, 1=u32, 2=i32
	uint offset_elements;
	uint flags;      // bit 0 = cell grid
	int clip_x;
	int clip_y;
	uint clip_w;
	uint clip_h;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];
[[vk::image_format("rgba8")]]
[[vk::binding(1, 0)]] RWTexture2D<float4> images[];

float load_value(uint index) {
	uint bits = heap[pc.values_idx].Load((pc.offset_elements + index) * 4u);
	if (pc.value_type == 1u) return float(bits);
	if (pc.value_type == 2u) return float(asint(bits));
	return asfloat(bits);
}

float3 viridis(float t) {
	// Compact polynomial approximation; monotonic luminance and color-blind
	// friendly without a texture/LUT dependency.
	float3 c0 = float3(0.277727, 0.005407, 0.334099);
	float3 c1 = float3(0.105093, 1.404613, 1.384590);
	float3 c2 = float3(-0.330861, 0.214847, 0.095095);
	float3 c3 = float3(-4.634230, -5.799100, -19.332440);
	float3 c4 = float3(6.228269, 14.179933, 56.690552);
	float3 c5 = float3(4.776384, -13.745145, -65.353032);
	float3 c6 = float3(-5.435455, 4.645852, 26.312435);
	return saturate(c0 + t * (c1 + t * (c2 + t * (c3 + t *
		(c4 + t * (c5 + t * c6))))));
}

float3 plasma(float t) {
	float3 a = float3(0.050, 0.030, 0.528);
	float3 b = float3(0.798, 0.280, 0.470);
	float3 c = float3(0.940, 0.975, 0.131);
	return t < 0.55 ? lerp(a, b, t / 0.55) : lerp(b, c, (t - 0.55) / 0.45);
}

float3 map_color(float t) {
	if (pc.colormap == 0u) return plasma(t);
	if (pc.colormap == 2u) {
		return t < 0.5
			? lerp(float3(0.230, 0.299, 0.754), float3(0.865), t * 2.0)
			: lerp(float3(0.865), float3(0.706, 0.016, 0.150), (t - 0.5) * 2.0);
	}
	if (pc.colormap == 3u) return float3(t);
	return viridis(t);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint x = tid.x;
	uint y = tid.y;
	if (x >= pc.dst_w || y >= pc.dst_h || pc.rows == 0u || pc.cols == 0u) return;
	int dst_x = pc.dst_x + int(x);
	int dst_y = pc.dst_y + int(y);
	if (dst_x < pc.clip_x || dst_y < pc.clip_y
		|| uint(dst_x - pc.clip_x) >= pc.clip_w
		|| uint(dst_y - pc.clip_y) >= pc.clip_h) return;
	uint image_width, image_height;
	images[pc.dst_idx].GetDimensions(image_width, image_height);
	if (dst_x < 0 || dst_y < 0
		|| dst_x >= int(image_width) || dst_y >= int(image_height)) return;

	uint col = min(pc.cols - 1u, x * pc.cols / pc.dst_w);
	uint row = min(pc.rows - 1u, y * pc.rows / pc.dst_h);
	float range = max(abs(pc.v_max - pc.v_min), 1.0e-12);
	float t = saturate((load_value(row * pc.cols + col) - pc.v_min) / range);
	float3 color = map_color(t);
	if ((pc.flags & 1u) != 0u) {
		uint cell_x = x * pc.cols / pc.dst_w;
		uint cell_y = y * pc.rows / pc.dst_h;
		uint next_x = min(pc.cols, (x + 1u) * pc.cols / pc.dst_w);
		uint next_y = min(pc.rows, (y + 1u) * pc.rows / pc.dst_h);
		if (next_x != cell_x || next_y != cell_y) color *= 0.65;
	}
	images[pc.dst_idx][uint2(uint(dst_x), uint(dst_y))] = float4(color, 1.0);
}
