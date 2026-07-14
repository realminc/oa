// Nv12ResizeNormalizeToBf16.slang — Fused NV12 → BF16 + Resize + Normalize
//
// Single-pass conversion from NV12 (YUV 4:2:0) to BF16 NCHW tensor with resize and normalization.
// Input:  NV12 VkImage (Y plane + UV plane)
// Output: [B, C, H_out, W_out] BF16 tensor
//
// Pipeline:
// 1. Sample Y and UV from NV12
// 2. Convert YUV to RGB
// 3. Resize via bilinear interpolation
// 4. Apply ImageNet normalization
//
// Dispatch: (ceil(W_out/16), ceil(H_out/16), B*C)

import Storage;

struct PushConstants {
    uint y_idx;         // Y plane buffer index
    uint uv_idx;        // UV plane buffer index
    uint out_idx;       // Output buffer index
    uint batch_size;    // B (always 1 for single image)
    uint h_in;          // Input height
    uint w_in;          // Input width
    uint h_out;         // Output height
    uint w_out;         // Output width
    float mean[3];      // Per-channel mean
    float std[3];       // Per-channel std
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

// Load Y component from NV12 Y plane
float LoadY(uint y, uint x) {
    if (y >= pc.h_in || x >= pc.w_in) return 0.0f;
    uint idx = y * pc.w_in + x;
    return OaLoad(heap[pc.y_idx], idx);
}

// Load UV components from NV12 UV plane (subsampled 2x2)
float2 LoadUV(uint y, uint x) {
    uint uv_h = pc.h_in / 2;
    uint uv_w = pc.w_in / 2;
    uint uv_y = y / 2;
    uint uv_x = x / 2;
    if (uv_y >= uv_h || uv_x >= uv_w) return float2(0.0f, 0.0f);
    uint idx = uv_y * uv_w + uv_x;
    // UV is interleaved: U and V in same byte (8-bit each)
    uint uv_packed = asuint(OaLoad(heap[pc.uv_idx], idx));
    return float2(float((uv_packed >> 0) & 0xFF), float((uv_packed >> 8) & 0xFF));
}

// YUV to RGB conversion (BT.601)
float3 YuvToRgb(float y, float u, float v) {
    // Normalize to [0, 1]
    float yn = y / 255.0f;
    float un = (u - 128.0f) / 255.0f;
    float vn = (v - 128.0f) / 255.0f;
    
    // BT.601 conversion matrix
    float r = yn + 1.402f * vn;
    float g = yn - 0.344136f * un - 0.714136f * vn;
    float b = yn + 1.772f * un;
    
    return float3(r, g, b);
}

// Bilinear interpolation for Y plane
float BilinearSampleY(float y_src, float x_src) {
    y_src = clamp(y_src, 0.0f, float(pc.h_in - 1));
    x_src = clamp(x_src, 0.0f, float(pc.w_in - 1));
    
    uint y0 = uint(floor(y_src));
    uint x0 = uint(floor(x_src));
    uint y1 = min(y0 + 1, pc.h_in - 1);
    uint x1 = min(x0 + 1, pc.w_in - 1);
    
    float fy = y_src - float(y0);
    float fx = x_src - float(x0);
    
    float v00 = LoadY(y0, x0);
    float v01 = LoadY(y0, x1);
    float v10 = LoadY(y1, x0);
    float v11 = LoadY(y1, x1);
    
    float v0 = lerp(v00, v01, fx);
    float v1 = lerp(v10, v11, fx);
    return lerp(v0, v1, fy);
}

// Nearest neighbor for UV (since UV is already subsampled)
float2 NearestSampleUV(float y_src, float x_src) {
    uint uv_h = pc.h_in / 2;
    uint uv_w = pc.w_in / 2;
    
    y_src = clamp(y_src / 2.0f, 0.0f, float(uv_h - 1));
    x_src = clamp(x_src / 2.0f, 0.0f, float(uv_w - 1));
    
    uint y = uint(round(y_src));
    uint x = uint(round(x_src));
    return LoadUV(y * 2, x * 2);
}

// Normalize value for channel c
float Normalize(float value, uint c) {
    float mean = (c < 3) ? pc.mean[c] : 0.0f;
    float std = (c < 3) ? pc.std[c] : 1.0f;
    return (value - mean) / std;
}

// Store to NCHW tensor [B, C, H, W]
void StoreNchw(uint buf_idx, uint b, uint c, uint h, uint w, uint C, uint H, uint W, float value) {
    uint idx = b * (C * H * W) + c * (H * W) + h * W + w;
    OaStore(heap[buf_idx], idx, value);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint x_out = gid.x * 16 + lid.x;
    uint y_out = gid.y * 16 + lid.y;
    uint bc = gid.z;  // Combined batch*channel index
    
    if (x_out >= pc.w_out || y_out >= pc.h_out) return;
    
    uint b = bc / 3;  // Always 3 channels (RGB)
    uint c = bc % 3;
    
    if (b >= pc.batch_size) return;
    
    // Compute source coordinates
    float scale_y = float(pc.h_in) / float(pc.h_out);
    float scale_x = float(pc.w_in) / float(pc.w_out);
    
    float y_src = (float(y_out) + 0.5f) * scale_y - 0.5f;
    float x_src = (float(x_out) + 0.5f) * scale_x - 0.5f;
    
    // Sample Y and UV
    float y = BilinearSampleY(y_src, x_src);
    float2 uv = NearestSampleUV(y_src, x_src);
    
    // Convert YUV to RGB
    float3 rgb = YuvToRgb(y, uv.x, uv.y);
    
    // Get the channel value
    float value = rgb[c];
    
    // Normalize
    value = Normalize(value, c);
    
    // Store result
    StoreNchw(pc.out_idx, b, c, y_out, x_out, 3, pc.h_out, pc.w_out, value);
}
