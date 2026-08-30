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
// Dispatch: (ceil(width/16), ceil(height/16), 1). Each 8x8 workgroup covers a
// 16x16 luma tile. One invocation owns one 2x2 luma block so the four pixels
// share their single 4:2:0 chroma sample.

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
    uint bit_depth;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(2, 0)]] Texture2D<float4> sampled_images[];
[[vk::binding(1, 0)]] RWTexture2D<float4> storage_images[];
[[vk::binding(3, 0)]] SamplerState samplers[];

struct ColorTransform {
    float luma_scale;
    float luma_offset;
    float3 u_coefficients;
    float3 v_coefficients;
};

ColorTransform getColorTransform() {
    ColorTransform transform;
    float code_maximum = float((1u << pc.bit_depth) - 1u);
    bool full_range = pc.full_range != 0;
    transform.luma_scale = full_range ? 1.0f : 1.164f;
    transform.luma_offset = full_range
        ? 0.0f
        : float(16u << (pc.bit_depth - 8u)) / code_maximum;

    if (pc.color_space == COLOR_SPACE_BT709) {
        transform.u_coefficients = full_range
            ? float3(0.0f, -0.1873f, 1.8556f)
            : float3(0.0f, -0.213f, 2.112f);
        transform.v_coefficients = full_range
            ? float3(1.5748f, -0.4681f, 0.0f)
            : float3(1.793f, -0.533f, 0.0f);
    } else if (pc.color_space == COLOR_SPACE_BT2020) {
        transform.u_coefficients = full_range
            ? float3(0.0f, -0.1646f, 1.8814f)
            : float3(0.0f, -0.188f, 2.148f);
        transform.v_coefficients = full_range
            ? float3(1.4746f, -0.5714f, 0.0f)
            : float3(1.678f, -0.650f, 0.0f);
    } else {
        transform.u_coefficients = full_range
            ? float3(0.0f, -0.344f, 1.772f)
            : float3(0.0f, -0.391f, 2.018f);
        transform.v_coefficients = full_range
            ? float3(1.402f, -0.714f, 0.0f)
            : float3(1.596f, -0.813f, 0.0f);
    }
    return transform;
}

void writePixel(
    uint x,
    uint y,
    ColorTransform transform,
    float3 chroma_contribution)
{
    if (x >= pc.width || y >= pc.height) return;

    // Sample Y (luma) at full resolution. The VkImage can be larger than the
    // visible frame because Vulkan Video sessions are aligned to coded extents.
    float2 y_coord = float2(x + 0.5f, y + 0.5f) / float2(pc.coded_width, pc.coded_height);
    float Y = sampled_images[pc.y_idx].SampleLevel(samplers[pc.sampler_idx], y_coord, 0).r;

    float3 rgb = saturate(
        transform.luma_scale * (Y - transform.luma_offset)
        + chroma_contribution);
    
    // Store RGB output
    storage_images[pc.out_idx][uint2(x, y)] = float4(rgb, 1.0f);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint x = gid.x * 16 + lid.x * 2;
    uint y = gid.y * 16 + lid.y * 2;
    if (x >= pc.width || y >= pc.height) return;

    // NV12/P010 4:2:0 stores exactly one interleaved chroma sample per 2x2
    // luma block. All four output pixels therefore use the same coordinate,
    // including with the existing nearest/linear sampler choices.
    float2 uv_coord = float2((x >> 1) + 0.5f, (y >> 1) + 0.5f)
        / float2(pc.coded_width >> 1, pc.coded_height >> 1);
    float2 UV = sampled_images[pc.uv_idx].SampleLevel(
        samplers[pc.sampler_idx], uv_coord, 0).rg;

    ColorTransform transform = getColorTransform();
    float chroma_midpoint = float(128u << (pc.bit_depth - 8u))
        / float((1u << pc.bit_depth) - 1u);
    float2 centered_chroma = UV - chroma_midpoint;
    float3 chroma_contribution =
        centered_chroma.x * transform.u_coefficients
        + centered_chroma.y * transform.v_coefficients;

    writePixel(x, y, transform, chroma_contribution);
    writePixel(x + 1, y, transform, chroma_contribution);
    writePixel(x, y + 1, transform, chroma_contribution);
    writePixel(x + 1, y + 1, transform, chroma_contribution);
}
