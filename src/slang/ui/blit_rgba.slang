// BlitRgba.slang — blit (and nearest-neighbour scale) an RGBA8 buffer region
// into the compose storage image.
//
// Donor: oacpp/source/cpp/lib/oa/ui/shader/compute/blitRgba.slang
// Adaptation: exact donor shader body; import path adjusted for Rust tree.
//
// Dispatch: (ceil(dst_w / 8), ceil(dst_h / 8), 1)

import storage;
#include "compositor_common.slang"

struct PushConstants {
	uint src_idx;    // bindless buffer index — RGBA8 pixels
	uint dst_idx;    // bindless image index  — compose storage image
	uint src_w;      // source image width  (pixels)
	uint src_h;      // source image height (pixels)
	int  dst_x;      // X offset in compose image
	int  dst_y;      // Y offset in compose image
	uint dst_w;      // destination rect width
	uint dst_h;      // destination rect height
	int  clip_x;
	int  clip_y;
	uint clip_w;
	uint clip_h;
	uint channel;
}

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];
[[vk::binding(1, 0)]] RWTexture2D<float4> images[];

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
	uint x = gid.x * 8 + lid.x;
	uint y = gid.y * 8 + lid.y;
	if (x >= pc.dst_w || y >= pc.dst_h || pc.src_w == 0 || pc.src_h == 0) return;

	int dstX = pc.dst_x + int(x);
	int dstY = pc.dst_y + int(y);
	if (dstX < pc.clip_x || dstY < pc.clip_y
		|| uint(dstX - pc.clip_x) >= pc.clip_w
		|| uint(dstY - pc.clip_y) >= pc.clip_h) return;
	uint imageWidth;
	uint imageHeight;
	images[pc.dst_idx].GetDimensions(imageWidth, imageHeight);
	if (dstX < 0 || dstY < 0
		|| dstX >= int(imageWidth) || dstY >= int(imageHeight)) return;

	// Nearest-neighbour scale.
	uint sx = x * pc.src_w / pc.dst_w;
	uint sy = y * pc.src_h / pc.dst_h;
	sx = min(sx, pc.src_w - 1);
	sy = min(sy, pc.src_h - 1);

	uint byteOffset = (sy * pc.src_w + sx) * 4;
	uint word = heap[pc.src_idx].Load(byteOffset);
	float r = float((word >>  0) & 0xffu) / 255.0;
	float g = float((word >>  8) & 0xffu) / 255.0;
	float b = float((word >> 16) & 0xffu) / 255.0;
	float a = float((word >> 24) & 0xffu) / 255.0;

	float4 src = float4(r, g, b, a);
	if (pc.channel != 0u) {
		float selected = pc.channel == 1u ? r : pc.channel == 2u ? g : pc.channel == 3u ? b : a;
		src = float4(selected, selected, selected, 1.0);
	}
	uint2 dstCoord = uint2(uint(dstX), uint(dstY));
	float4 dst = images[pc.dst_idx][dstCoord];
	images[pc.dst_idx][dstCoord] = uiSourceOver(src, dst);
}
