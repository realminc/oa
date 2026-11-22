// Raster adaptation of cvt_yuv420_to_rgba8.slang and blit_texture.frag.slang.
// Keeps 10/12-bit decoder planes packed until the canvas draw.
import yuv_to_rgb;

struct PushConstants {
	uint src_w;
	uint src_h;
	int dst_x;
	int dst_y;
	uint dst_w;
	uint dst_h;
	uint cb_offset;
	uint cr_offset;
	uint sample_bits;
	uint interleaved;
	uint color_space;
	uint full_range;
};
[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] ByteAddressBuffer heap[1];

uint load_sample(uint offset) {
	uint word = heap[0].Load(offset & ~3u);
	uint shift = (offset & 2u) * 8u;
	return ((word >> shift) & 0xffffu) >> (16u - pc.sample_bits);
}

float3 texel(uint2 pixel) {
	uint chroma_width = (pc.src_w + 1u) >> 1;
	uint2 chroma = pixel >> 1;
	uint chroma_index = chroma.y * chroma_width + chroma.x;
	uint y = load_sample((pixel.y * pc.src_w + pixel.x) * 2u);
	uint cb;
	uint cr;
	if (pc.interleaved != 0u) {
		uint offset = pc.cb_offset + chroma_index * 4u;
		cb = load_sample(offset);
		cr = load_sample(offset + 2u);
	} else {
		cb = load_sample(pc.cb_offset + chroma_index * 2u);
		cr = load_sample(pc.cr_offset + chroma_index * 2u);
	}
	float3 yuv;
	float shift = float(1u << (pc.sample_bits - 8u));
	if (pc.full_range != 0u) {
		float maximum = float((1u << pc.sample_bits) - 1u);
		yuv = float3(y, float(cb) - 128.0f * shift, float(cr) - 128.0f * shift) / maximum;
	} else {
		yuv = (float3(y, cb, cr) - float3(16.0f, 128.0f, 128.0f) * shift)
			/ (float3(219.0f, 224.0f, 224.0f) * shift);
	}
	return round(yuv_to_rgb(yuv, pc.color_space) * 255.0f) / 255.0f;
}

[shader("fragment")]
float4 main(float4 position : SV_Position) : SV_Target0 {
	uint2 local = uint2(int2(position.xy) - int2(pc.dst_x, pc.dst_y));
	uint2 source = uint2(pc.src_w, pc.src_h);
	uint2 destination = uint2(pc.dst_w, pc.dst_h);
	if (any(local >= destination)) discard;

	if (all(destination >= source)) {
		uint2 pixel = min(local * source / destination, source - 1u);
		return float4(texel(pixel), 1.0f);
	}
	// Match the RGBA path's bilinear sampling including per-texel quantization.
	float2 sample = clamp((float2(local) + 0.5f) * float2(source)
		/ float2(destination) - 0.5f, 0.0f, float2(source - 1u));
	uint2 a = uint2(sample);
	uint2 b = min(a + 1u, source - 1u);
	float2 weight = frac(sample);
	float3 top = lerp(texel(a), texel(uint2(b.x, a.y)), weight.x);
	float3 bottom = lerp(texel(uint2(a.x, b.y)), texel(b), weight.x);
	return float4(lerp(top, bottom, weight.y), 1.0f);
}
