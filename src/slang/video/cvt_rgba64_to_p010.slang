// Packed RGBA64 buffer -> 16-bit two-plane 4:2:0 storage image for the
// encoder (P010/G10X6: 10-bit samples held in the top bits of 16-bit UNORM
// plane views, so normalized [0,1] writes quantize correctly).
//
// Sibling of cvt_rgba_to_nv12.slang for wide-component encode profiles.
// Reads the encoder's RGBA64 source from the bindless storage-buffer heap
// and writes the encoder's input picture:
//   - Y plane at full resolution  (R16_UNORM view, plane 0)
//   - UV plane at half resolution (R16G16_UNORM view, plane 1), interleaved Cb/Cr
//
// Dispatch: (ceil(coded_w/16), ceil(coded_h/16), 1)
//   - Each 16x16 thread group reads a 16x16 region of RGBA64, writes 16x16
//     Y values, and writes 8x8 UV values (each UV computed as the
//     average of the corresponding 2x2 RGB block).

import rgb_to_yuv;

struct PushConstants {
    uint rgba_idx;       // bindless heap[] index for source RGBA64 buffer
    uint y_idx;          // bindless storage_images[] index for Y-plane view  (R16_UNORM)
    uint uv_idx;         // bindless storage_images[] index for UV-plane view (R16G16_UNORM)
    uint width;          // visible width
    uint height;         // visible height
    uint coded_width;    // aligned (encoder coded) width — pad regions get 0
    uint coded_height;   // aligned (encoder coded) height
    uint color_space;
    uint full_range;     // 0 = limited (TV), 1 = full (PC)
    uint _pad;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];
[[vk::binding(1, 0)]] RWTexture2D<float4> storage_images[];

// Loads a packed RGBA64 pair from heap[rgba_idx] at pixel (px) and unpacks
// the 16-bit channels to normalized [0,1] floats. Pixels outside the
// visible frame return black so encoder pad regions don't read into
// uninitialised storage.
float4 load_rgba_safe(uint2 px) {
    if (px.x >= pc.width || px.y >= pc.height) return float4(0, 0, 0, 1);
    uint byte_offset = (px.y * pc.width + px.x) * 8;
    uint2 words = heap[pc.rgba_idx].Load2(byte_offset);
    float r = float(words.x & 0xffff) / 65535.0f;
    float g = float(words.x >> 16) / 65535.0f;
    float b = float(words.y & 0xffff) / 65535.0f;
    float a = float(words.y >> 16) / 65535.0f;
    return float4(r, g, b, a);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint x = gid.x * 16 + lid.x;
    uint y = gid.y * 16 + lid.y;

    // ── Y plane (full resolution) ──────────────────────────────────────
    if (x < pc.coded_width && y < pc.coded_height) {
        float4 rgba = load_rgba_safe(uint2(x, y));
        float3 yuv = rgb_to_yuv(rgba.rgb, pc.color_space, pc.full_range);
        storage_images[pc.y_idx][uint2(x, y)] = float4(yuv.x, 0, 0, 1);
    }

    // ── UV plane (half resolution, 4:2:0 subsampling) ──────────────────
    if ((lid.x & 1U) == 0U && (lid.y & 1U) == 0U) {
        uint cx = x >> 1;
        uint cy = y >> 1;
        uint ccw = pc.coded_width >> 1;
        uint cch = pc.coded_height >> 1;
        if (cx < ccw && cy < cch) {
            float4 r00 = load_rgba_safe(uint2(x, y));
            float4 r10 = load_rgba_safe(uint2(x + 1, y));
            float4 r01 = load_rgba_safe(uint2(x, y + 1));
            float4 r11 = load_rgba_safe(uint2(x + 1, y + 1));
            float3 rgb_avg = (r00.rgb + r10.rgb + r01.rgb + r11.rgb) * 0.25f;
            float3 yuv = rgb_to_yuv(rgb_avg, pc.color_space, pc.full_range);
            storage_images[pc.uv_idx][uint2(cx, cy)] = float4(yuv.y, yuv.z, 0, 1);
        }
    }
}
