// Sampled RGBA/BGRA VkImage -> NV12 multi-plane storage image.
// Texture2D.Load applies the source view's format mapping, so both
// R8G8B8A8_UNORM and B8G8R8A8_UNORM views produce logical RGBA values.

import RgbToYuv;

struct PushConstants {
    uint rgba_idx;
    uint y_idx;
    uint uv_idx;
    uint width;
    uint height;
    uint coded_width;
    uint coded_height;
    uint color_space;
    uint full_range;
    uint _pad;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(1, 0)]] RWTexture2D<float4> storage_images[];
[[vk::binding(2, 0)]] Texture2D<float4> sampled_images[];

float4 LoadRgbaSafe(uint2 px) {
    if (px.x >= pc.width || px.y >= pc.height) return float4(0, 0, 0, 1);
    return sampled_images[pc.rgba_idx].Load(int3(int(px.x), int(px.y), 0));
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint x = gid.x * 16 + lid.x;
    uint y = gid.y * 16 + lid.y;

    if (x < pc.coded_width && y < pc.coded_height) {
        float3 yuv = RgbToYuv(LoadRgbaSafe(uint2(x, y)).rgb, pc.color_space, pc.full_range);
        storage_images[pc.y_idx][uint2(x, y)] = float4(yuv.x, 0, 0, 1);
    }

    if ((lid.x & 1U) == 0U && (lid.y & 1U) == 0U) {
        uint cx = x >> 1;
        uint cy = y >> 1;
        if (cx < (pc.coded_width >> 1) && cy < (pc.coded_height >> 1)) {
            float3 rgb = (LoadRgbaSafe(uint2(x, y)).rgb
                + LoadRgbaSafe(uint2(x + 1, y)).rgb
                + LoadRgbaSafe(uint2(x, y + 1)).rgb
                + LoadRgbaSafe(uint2(x + 1, y + 1)).rgb) * 0.25f;
            float3 yuv = RgbToYuv(rgb, pc.color_space, pc.full_range);
            storage_images[pc.uv_idx][uint2(cx, cy)] = float4(yuv.y, yuv.z, 0, 1);
        }
    }
}
