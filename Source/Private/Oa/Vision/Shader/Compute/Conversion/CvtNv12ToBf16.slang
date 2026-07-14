// CvtNv12ToBf16.slang — NV12 → RGB → BF16 Conversion
//
// Zero-copy video decode → ML pipeline:
// - Input: VkImage (NV12 format: Y plane + interleaved UV plane)
// - Output: BF16 tensor [B, 3, H, W] (NCHW layout)
// - Optional: ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
//
// NV12 Layout:
// - Y plane: width × height (luma, full resolution)
// - UV plane: (width/2) × (height/2) (chroma, half resolution, interleaved U/V)
//
// YUV → RGB conversion (BT.601 limited range):
// R = 1.164 * (Y - 16) + 1.596 * (V - 128)
// G = 1.164 * (Y - 16) - 0.391 * (U - 128) - 0.813 * (V - 128)
// B = 1.164 * (Y - 16) + 2.018 * (U - 128)
//
// Dispatch: (ceil(width/16), ceil(height/16), batch_size)

import Storage;

struct PushConstants {
    uint out_idx;       // Output buffer index (BF16 tensor)
    uint y_idx;         // Sampled Y plane image index
    uint uv_idx;        // Sampled UV plane image index
    uint sampler_idx;   // Sampler index
    uint width;         // Image width
    uint height;        // Image height
    uint batch_idx;     // Current batch index
    uint normalize;     // 1 = apply ImageNet normalization, 0 = raw [0,1]
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];
[[vk::binding(2, 0)]] Texture2D<float4> sampled_images[];
[[vk::binding(3, 0)]] SamplerState samplers[];

// ImageNet normalization constants
static const float3 IMAGENET_MEAN = float3(0.485f, 0.456f, 0.406f);
static const float3 IMAGENET_STD  = float3(0.229f, 0.224f, 0.225f);

// YUV → RGB conversion (BT.601 limited range)
float3 YuvToRgb(float y, float u, float v) {
    float Y = 1.164f * (y - 16.0f / 255.0f);
    float U = u - 128.0f / 255.0f;
    float V = v - 128.0f / 255.0f;
    
    float r = Y + 1.596f * V;
    float g = Y - 0.391f * U - 0.813f * V;
    float b = Y + 2.018f * U;
    
    return saturate(float3(r, g, b));
}

// Store RGB value to NCHW tensor [B, C, H, W]
void StoreRgbNchw(uint batch, uint c, uint h, uint w, float value) {
    uint C = 3;
    uint H = pc.height;
    uint W = pc.width;
    uint idx = batch * (C * H * W) + c * (H * W) + h * W + w;
    OaStore(heap[pc.out_idx], idx, value);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint x = gid.x * 16 + lid.x;
    uint y = gid.y * 16 + lid.y;
    uint batch = gid.z;
    
    if (x >= pc.width || y >= pc.height) return;
    
    // Sample Y (luma) at full resolution
    float2 y_coord = float2(x + 0.5f, y + 0.5f) / float2(pc.width, pc.height);
    float Y = sampled_images[pc.y_idx].SampleLevel(samplers[pc.sampler_idx], y_coord, 0).r;
    
    // Sample UV (chroma) at half resolution. One UV texel covers a 2x2 luma
    // block, so address chroma texel centers explicitly instead of reusing the
    // luma coordinate.
    float2 uv_coord = float2((x >> 1) + 0.5f, (y >> 1) + 0.5f)
        / float2(pc.width >> 1, pc.height >> 1);
    float2 UV = sampled_images[pc.uv_idx].SampleLevel(samplers[pc.sampler_idx], uv_coord, 0).rg;
    
    // Convert YUV → RGB
    float3 rgb = YuvToRgb(Y, UV.r, UV.g);
    
    // Apply ImageNet normalization if requested
    if (pc.normalize != 0) {
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD;
    }
    
    // Store as BF16 in NCHW layout [B, 3, H, W]
    StoreRgbNchw(batch, 0, y, x, rgb.r);  // R channel
    StoreRgbNchw(batch, 1, y, x, rgb.g);  // G channel
    StoreRgbNchw(batch, 2, y, x, rgb.b);  // B channel
}
