// CvtNv12ToRgb.slang - NV12 -> RGBA conversion with color space selection.
//
// Supports multiple color space standards:
// - BT.601: Standard Definition (SD) video
// - BT.709: High Definition (HD) video - 720p, 1080p
// - BT.2020: Ultra High Definition (UHD) video - 4K, 8K, HDR
//
// NV12 Layout:
// - Y plane: width × height (luma, full resolution)
// - UV plane: (width/2) × (height/2) (chroma, half resolution, interleaved U/V)
//
// Dispatch: (ceil(width/16), ceil(height/16), 1)

// Color space selection
static const uint COLOR_SPACE_BT601  = 0;
static const uint COLOR_SPACE_BT709  = 1;
static const uint COLOR_SPACE_BT2020 = 2;

struct PushConstants {
    uint y_idx;
    uint uv_idx;
    uint out_idx;
    uint sampler_idx;
    uint width;
    uint height;
    uint coded_width;
    uint coded_height;
    uint color_space;
    uint full_range;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(2, 0)]] Texture2D<float4> sampled_images[];
[[vk::binding(1, 0)]] RWTexture2D<float4> storage_images[];
[[vk::binding(3, 0)]] SamplerState samplers[];

// BT.601 (Standard Definition) - Limited Range
// R = 1.164 * (Y - 16) + 1.596 * (V - 128)
// G = 1.164 * (Y - 16) - 0.391 * (U - 128) - 0.813 * (V - 128)
// B = 1.164 * (Y - 16) + 2.018 * (U - 128)
float3 YuvToRgb_BT601_Limited(float y, float u, float v) {
    float Y = 1.164f * (y - 16.0f / 255.0f);
    float U = u - 128.0f / 255.0f;
    float V = v - 128.0f / 255.0f;
    
    float r = Y + 1.596f * V;
    float g = Y - 0.391f * U - 0.813f * V;
    float b = Y + 2.018f * U;
    
    return saturate(float3(r, g, b));
}

// BT.601 (Standard Definition) - Full Range
float3 YuvToRgb_BT601_Full(float y, float u, float v) {
    float Y = y;
    float U = u - 0.5f;
    float V = v - 0.5f;
    
    float r = Y + 1.402f * V;
    float g = Y - 0.344f * U - 0.714f * V;
    float b = Y + 1.772f * U;
    
    return saturate(float3(r, g, b));
}

// BT.709 (High Definition) - Limited Range
// R = 1.164 * (Y - 16) + 1.793 * (V - 128)
// G = 1.164 * (Y - 16) - 0.213 * (U - 128) - 0.533 * (V - 128)
// B = 1.164 * (Y - 16) + 2.112 * (U - 128)
float3 YuvToRgb_BT709_Limited(float y, float u, float v) {
    float Y = 1.164f * (y - 16.0f / 255.0f);
    float U = u - 128.0f / 255.0f;
    float V = v - 128.0f / 255.0f;
    
    float r = Y + 1.793f * V;
    float g = Y - 0.213f * U - 0.533f * V;
    float b = Y + 2.112f * U;
    
    return saturate(float3(r, g, b));
}

// BT.709 (High Definition) - Full Range
float3 YuvToRgb_BT709_Full(float y, float u, float v) {
    float Y = y;
    float U = u - 0.5f;
    float V = v - 0.5f;
    
    float r = Y + 1.5748f * V;
    float g = Y - 0.1873f * U - 0.4681f * V;
    float b = Y + 1.8556f * U;
    
    return saturate(float3(r, g, b));
}

// BT.2020 (Ultra High Definition) - Limited Range
// R = 1.164 * (Y - 16) + 1.678 * (V - 128)
// G = 1.164 * (Y - 16) - 0.188 * (U - 128) - 0.650 * (V - 128)
// B = 1.164 * (Y - 16) + 2.148 * (U - 128)
float3 YuvToRgb_BT2020_Limited(float y, float u, float v) {
    float Y = 1.164f * (y - 16.0f / 255.0f);
    float U = u - 128.0f / 255.0f;
    float V = v - 128.0f / 255.0f;
    
    float r = Y + 1.678f * V;
    float g = Y - 0.188f * U - 0.650f * V;
    float b = Y + 2.148f * U;
    
    return saturate(float3(r, g, b));
}

// BT.2020 (Ultra High Definition) - Full Range
float3 YuvToRgb_BT2020_Full(float y, float u, float v) {
    float Y = y;
    float U = u - 0.5f;
    float V = v - 0.5f;
    
    float r = Y + 1.4746f * V;
    float g = Y - 0.1646f * U - 0.5714f * V;
    float b = Y + 1.8814f * U;
    
    return saturate(float3(r, g, b));
}

// Router function for color space conversion
float3 YuvToRgb(float y, float u, float v, uint color_space, uint full_range) {
    if (color_space == COLOR_SPACE_BT709) {
        return full_range != 0 
            ? YuvToRgb_BT709_Full(y, u, v)
            : YuvToRgb_BT709_Limited(y, u, v);
    }
    else if (color_space == COLOR_SPACE_BT2020) {
        return full_range != 0
            ? YuvToRgb_BT2020_Full(y, u, v)
            : YuvToRgb_BT2020_Limited(y, u, v);
    }
    else { // BT.601 (default)
        return full_range != 0
            ? YuvToRgb_BT601_Full(y, u, v)
            : YuvToRgb_BT601_Limited(y, u, v);
    }
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint x = gid.x * 16 + lid.x;
    uint y = gid.y * 16 + lid.y;

    if (x >= pc.width || y >= pc.height) return;

    // Sample Y (luma) at full resolution. The VkImage can be larger than the
    // visible frame because Vulkan Video sessions are aligned to coded extents.
    float2 y_coord = float2(x + 0.5f, y + 0.5f) / float2(pc.coded_width, pc.coded_height);
    float Y = sampled_images[pc.y_idx].SampleLevel(samplers[pc.sampler_idx], y_coord, 0).r;

    // Sample UV (chroma) at half resolution. One UV texel covers a 2x2 luma
    // block, so address chroma texel centers explicitly instead of reusing the
    // luma coordinate.
    float2 uv_coord = float2((x >> 1) + 0.5f, (y >> 1) + 0.5f)
        / float2(pc.coded_width >> 1, pc.coded_height >> 1);
    float2 UV = sampled_images[pc.uv_idx].SampleLevel(samplers[pc.sampler_idx], uv_coord, 0).rg;

    // Convert YUV -> RGB with selected color space
    float3 rgb = YuvToRgb(Y, UV.r, UV.g, pc.color_space, pc.full_range);
    
    // Store RGB output
    storage_images[pc.out_idx][uint2(x, y)] = float4(rgb, 1.0f);
}
