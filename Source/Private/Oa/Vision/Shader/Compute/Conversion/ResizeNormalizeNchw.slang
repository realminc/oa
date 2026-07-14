// ResizeNormalizeNchw.slang — Fused Resize + Normalize
//
// Combines bilinear resize and ImageNet-style normalization in a single kernel.
// Input:  [B, C, H_in, W_in]  (BF16 or F32)
// Output: [B, C, H_out, W_out] (BF16 or F32)
//
// Normalization: (x - mean) / std per channel
// Resize: Bilinear interpolation
//
// Dispatch: (ceil(W_out/16), ceil(H_out/16), B*C)

import Storage;

struct PushConstants {
    uint in_idx;        // Input buffer index
    uint out_idx;       // Output buffer index
    uint batch_size;    // B
    uint channels;      // C
    uint h_in;          // Input height
    uint w_in;          // Input width
    uint h_out;         // Output height
    uint w_out;         // Output width
    float mean[3];      // Per-channel mean
    float std[3];       // Per-channel std
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

// Load from NCHW tensor [B, C, H, W]
float LoadNchw(uint buf_idx, uint b, uint c, uint h, uint w, uint C, uint H, uint W) {
    if (h >= H || w >= W) return 0.0f;
    uint idx = b * (C * H * W) + c * (H * W) + h * W + w;
    return OaLoad(heap[buf_idx], idx);
}

// Store to NCHW tensor [B, C, H, W]
void StoreNchw(uint buf_idx, uint b, uint c, uint h, uint w, uint C, uint H, uint W, float value) {
    uint idx = b * (C * H * W) + c * (H * W) + h * W + w;
    OaStore(heap[buf_idx], idx, value);
}

// Bilinear interpolation
float BilinearSample(uint buf_idx, uint b, uint c, float y_src, float x_src) {
    // Clamp to valid range
    y_src = clamp(y_src, 0.0f, float(pc.h_in - 1));
    x_src = clamp(x_src, 0.0f, float(pc.w_in - 1));
    
    // Get integer coordinates and fractional parts
    uint y0 = uint(floor(y_src));
    uint x0 = uint(floor(x_src));
    uint y1 = min(y0 + 1, pc.h_in - 1);
    uint x1 = min(x0 + 1, pc.w_in - 1);
    
    float fy = y_src - float(y0);
    float fx = x_src - float(x0);
    
    // Sample 4 neighbors
    float v00 = LoadNchw(buf_idx, b, c, y0, x0, pc.channels, pc.h_in, pc.w_in);
    float v01 = LoadNchw(buf_idx, b, c, y0, x1, pc.channels, pc.h_in, pc.w_in);
    float v10 = LoadNchw(buf_idx, b, c, y1, x0, pc.channels, pc.h_in, pc.w_in);
    float v11 = LoadNchw(buf_idx, b, c, y1, x1, pc.channels, pc.h_in, pc.w_in);
    
    // Bilinear interpolation
    float v0 = lerp(v00, v01, fx);
    float v1 = lerp(v10, v11, fx);
    return lerp(v0, v1, fy);
}

// Normalize value for channel c
float Normalize(float value, uint c) {
    float mean = (c < 3) ? pc.mean[c] : 0.0f;
    float std = (c < 3) ? pc.std[c] : 1.0f;
    return (value - mean) / std;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint x_out = gid.x * 16 + lid.x;
    uint y_out = gid.y * 16 + lid.y;
    uint bc = gid.z;  // Combined batch*channel index
    
    if (x_out >= pc.w_out || y_out >= pc.h_out) return;
    
    uint b = bc / pc.channels;
    uint c = bc % pc.channels;
    
    if (b >= pc.batch_size) return;
    
    // Compute source coordinates (align_corners=False, PyTorch default)
    float scale_y = float(pc.h_in) / float(pc.h_out);
    float scale_x = float(pc.w_in) / float(pc.w_out);
    
    float y_src = (float(y_out) + 0.5f) * scale_y - 0.5f;
    float x_src = (float(x_out) + 0.5f) * scale_x - 0.5f;
    
    // Bilinear sample
    float value = BilinearSample(pc.in_idx, b, c, y_src, x_src);
    
    // Normalize
    value = Normalize(value, c);
    
    // Store result
    StoreNchw(pc.out_idx, b, c, y_out, x_out, pc.channels, pc.h_out, pc.w_out, value);
}
