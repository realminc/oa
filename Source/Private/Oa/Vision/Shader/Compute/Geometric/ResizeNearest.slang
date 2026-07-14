// ResizeNearest.slang — Nearest Neighbor Image Resize
//
// Resizes BF16 tensor [B, C, H_in, W_in] → [B, C, H_out, W_out] using nearest neighbor.
// Faster than bilinear, suitable for integer scaling factors or when quality is not critical.
//
// Nearest neighbor:
// - For each output pixel (x_out, y_out), compute source coordinates:
//   x_src = floor(x_out * (W_in / W_out))
//   y_src = floor(y_out * (H_in / H_out))
// - Sample single nearest pixel (no interpolation)
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
    
    // OpenCV INTER_NEAREST / PyTorch nearest convention.
    float scale_y = float(pc.h_in) / float(pc.h_out);
    float scale_x = float(pc.w_in) / float(pc.w_out);

    uint y_src = min(uint(floor(float(y_out) * scale_y)), pc.h_in - 1);
    uint x_src = min(uint(floor(float(x_out) * scale_x)), pc.w_in - 1);
    
    // Load single pixel
    float value = LoadNchw(pc.in_idx, b, c, y_src, x_src, pc.channels, pc.h_in, pc.w_in);
    
    // Store result
    StoreNchw(pc.out_idx, b, c, y_out, x_out, pc.channels, pc.h_out, pc.w_out, value);
}
