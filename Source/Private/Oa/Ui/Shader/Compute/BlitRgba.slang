// BlitRgba.slang — blit (and scale) an RGBA8 buffer region into a storage image.
//
// Source:      bindless RGBA8 buffer (4 bytes per pixel, row-major, sRGB encoded).
// Destination: bindless storage image (linear float4, R8G8B8A8_SRGB or equivalent).
//
// If dst_w != src_w or dst_h != src_h the source is nearest-neighbour scaled to
// fill the destination rect.  Dispatch over dst_w × dst_h output pixels.
//
// Source bytes are sRGB-encoded; compose image is UNORM so we keep them as-is.
//
// Dispatch: (ceil(dst_w / 8), ceil(dst_h / 8), 1)

import Storage;

struct PushConstants {
    uint src_idx;  // bindless buffer index  — RGBA8 pixels
    uint dst_idx;  // bindless image index   — compose storage image
    uint src_w;    // source image width  (pixels)
    uint src_h;    // source image height (pixels)
    uint dst_x;    // X offset in compose image (pixels)
    uint dst_y;    // Y offset in compose image (pixels)
    uint dst_w;    // destination rect width  (pixels)
    uint dst_h;    // destination rect height (pixels)
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];
[[vk::binding(1, 0)]] RWTexture2D<float4> images[];

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint x = gid.x * 8 + lid.x;
    uint y = gid.y * 8 + lid.y;
    if (x >= pc.dst_w || y >= pc.dst_h) return;

    // Map output pixel to source pixel (nearest-neighbour).
    uint sx = x * pc.src_w / pc.dst_w;
    uint sy = y * pc.src_h / pc.dst_h;
    sx = min(sx, pc.src_w - 1);
    sy = min(sy, pc.src_h - 1);

    uint byte_offset = (sy * pc.src_w + sx) * 4;
    uint word = heap[pc.src_idx].Load(byte_offset);

    float r = float((word >>  0) & 0xFF) / 255.0f;
    float g = float((word >>  8) & 0xFF) / 255.0f;
    float b = float((word >> 16) & 0xFF) / 255.0f;
    float a = float((word >> 24) & 0xFF) / 255.0f;

    images[pc.dst_idx][uint2(pc.dst_x + x, pc.dst_y + y)] = float4(r, g, b, a);
}
