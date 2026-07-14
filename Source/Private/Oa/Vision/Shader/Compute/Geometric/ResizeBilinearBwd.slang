// ResizeBilinearBwd.slang — Backward pass for Bilinear Upsampling
//
// Input: grad_output [B, C, H_out, W_out]
// Output: grad_input [B, C, H_in, W_in] (pre-initialized to zeros)
// Each output pixel's gradient is distributed to 4 input pixels with
// the same bilinear weights used in the forward pass.
//
// Uses OaAtomicAdd for accumulation. The output buffer must be zero-initialized.
// Dispatch: (ceil(W_out/16), ceil(H_out/16), B*C)

import Storage;

struct PushConstants {
    uint in_idx;        // grad_output buffer index
    uint out_idx;       // grad_input buffer index (pre-initialized to zeros)
    uint batch_size;
    uint channels;
    uint h_in;
    uint w_in;
    uint h_out;
    uint w_out;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

float LoadNchw(uint buf_idx, uint b, uint c, uint h, uint w, uint C, uint H, uint W) {
    if (h >= H || w >= W) return 0.0f;
    uint idx = b * (C * H * W) + c * (H * W) + h * W + w;
    return OaLoad(heap[buf_idx], idx);
}

void AtomicAddNchw(uint buf_idx, uint b, uint c, uint h, uint w, uint C, uint H, uint W, float value) {
    if (h >= H || w >= W) return;
    uint idx = b * (C * H * W) + c * (H * W) + h * W + w;
    OaAtomicAdd(heap[buf_idx], idx, value);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint x_out = gid.x * 16 + lid.x;
    uint y_out = gid.y * 16 + lid.y;
    uint bc = gid.z;

    if (x_out >= pc.w_out || y_out >= pc.h_out) return;

    uint b = bc / pc.channels;
    uint c = bc % pc.channels;
    if (b >= pc.batch_size) return;

    float scale_y = float(pc.h_in) / float(pc.h_out);
    float scale_x = float(pc.w_in) / float(pc.w_out);

    float y_src = (float(y_out) + 0.5f) * scale_y - 0.5f;
    float x_src = (float(x_out) + 0.5f) * scale_x - 0.5f;

    int y0 = int(floor(y_src));
    int x0 = int(floor(x_src));
    int y1 = y0 + 1;
    int x1 = x0 + 1;

    float fy = y_src - float(y0);
    float fx = x_src - float(x0);

    y0 = clamp(y0, 0, int(pc.h_in) - 1);
    x0 = clamp(x0, 0, int(pc.w_in) - 1);
    y1 = clamp(y1, 0, int(pc.h_in) - 1);
    x1 = clamp(x1, 0, int(pc.w_in) - 1);

    float w00 = (1.0f - fy) * (1.0f - fx);
    float w01 = (1.0f - fy) * fx;
    float w10 = fy * (1.0f - fx);
    float w11 = fy * fx;

    float grad_out = LoadNchw(pc.in_idx, b, c, y_out, x_out, pc.channels, pc.h_out, pc.w_out);

    AtomicAddNchw(pc.out_idx, b, c, uint(y0), uint(x0), pc.channels, pc.h_in, pc.w_in, grad_out * w00);
    AtomicAddNchw(pc.out_idx, b, c, uint(y0), uint(x1), pc.channels, pc.h_in, pc.w_in, grad_out * w01);
    AtomicAddNchw(pc.out_idx, b, c, uint(y1), uint(x0), pc.channels, pc.h_in, pc.w_in, grad_out * w10);
    AtomicAddNchw(pc.out_idx, b, c, uint(y1), uint(x1), pc.channels, pc.h_in, pc.w_in, grad_out * w11);
}
