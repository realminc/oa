// Packed-planar Y'CbCr 4:2:0 -> RGBA8 presentation conversion.
//
// Companion to cvt_rgba_to_nv12.slang. The decode session copies one picture
// into a tightly packed buffer (full-resolution luma followed by interleaved
// NV12 chroma or two I420-style chroma planes); this kernel expands each
// visible pixel to one little-endian RGBA8 word for Texture upload. Chroma
// upsampling is nearest replication, matching the readback normalization
// contract. Both 8-bit and packed-16 10/12-bit plane storage are handled.

import yuv_to_rgb;
import storage;

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

struct CvtYuvToRgba8Push {
	uint src_idx;
	uint dst_idx;
	uint src_width;
	uint src_height;
	uint dst_width;
	uint dst_height;
	uint uv_offset;
	uint v_offset;
	uint uv_layout;
	uint color_space;
	uint full_range;
	uint sample_bits;
};
[[vk::push_constant]] CvtYuvToRgba8Push pc;

static const uint UV_LAYOUT_NV12 = 0;
static const uint UV_LAYOUT_PLANAR = 1;

uint load_sample(uint byte_offset) {
	// ByteAddressBuffer.Load addresses a 32-bit word. Extract the byte or
	// aligned halfword explicitly; passing a component address to Load loses
	// its low address bits on SPIR-V backends.
	if (pc.sample_bits <= 8) {
		return load_u8(heap[pc.src_idx], byte_offset);
	}
	uint word = heap[pc.src_idx].Load(byte_offset & ~3u);
	uint shift = (byte_offset & 3u) * 8u;
	uint raw = (word >> shift) & 0xffffu;
	return raw >> (16u - pc.sample_bits);
}

float scale_luma(uint v) {
	if (pc.full_range != 0) {
		return float(v) / float((1u << pc.sample_bits) - 1u);
	}
	float shift = float(1u << (pc.sample_bits - 8u));
	// Preserve foot/headroom until the final RGB clamp, as Vulkan YCbCr
	// conversion does. Clamping Y first changes saturated colors near black.
	return (float(v) - 16.0f * shift) / (219.0f * shift);
}

float scale_chroma(uint v) {
	if (pc.full_range != 0) {
		return (float(v) - float(1u << (pc.sample_bits - 1u))) / float((1u << pc.sample_bits) - 1u);
	}
	float shift = float(1u << (pc.sample_bits - 8u));
	return (float(v) - 128.0f * shift) / (224.0f * shift);
}

[numthreads(16, 16, 1)]
[shader("compute")]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
	uint x = gid.x * 16 + lid.x;
	uint y = gid.y * 16 + lid.y;
	if (x >= pc.dst_width || y >= pc.dst_height) {
		return;
	}
	uint sample_bytes = (pc.sample_bits <= 8) ? 1u : 2u;
	uint chroma_width = (pc.src_width + 1u) >> 1;
	uint chroma_height = (pc.src_height + 1u) >> 1;
	uint cx = min(x >> 1, chroma_width - 1u);
	uint cy = min(y >> 1, chroma_height - 1u);
	uint chroma_site = cy * chroma_width + cx;
	uint luma = load_sample((y * pc.src_width + x) * sample_bytes);
	uint cb;
	uint cr;
	if (pc.uv_layout == UV_LAYOUT_PLANAR) {
		cb = load_sample(pc.uv_offset + chroma_site * sample_bytes);
		cr = load_sample(pc.v_offset + chroma_site * sample_bytes);
	} else {
		cb = load_sample(pc.uv_offset + chroma_site * sample_bytes * 2u);
		cr = load_sample(pc.uv_offset + chroma_site * sample_bytes * 2u + sample_bytes);
	}
	float3 rgb = yuv_to_rgb(
		float3(scale_luma(luma), scale_chroma(cb), scale_chroma(cr)),
		pc.color_space
	);
	uint r = uint(round(saturate(rgb.r) * 255.0f));
	uint g = uint(round(saturate(rgb.g) * 255.0f));
	uint b = uint(round(saturate(rgb.b) * 255.0f));
	uint word = r | (g << 8) | (b << 16) | (0xffu << 24);
	heap[pc.dst_idx].Store((y * pc.dst_width + x) * 4u, word);
}
