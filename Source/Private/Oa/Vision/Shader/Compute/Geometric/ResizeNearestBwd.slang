// ResizeNearestBwd.slang — Backward pass for Nearest Neighbor Upsampling
//
// Input: grad_output [B, C, H_out, W_out]
// Output: grad_input [B, C, H_in, W_in]
// Each input pixel receives the sum of gradients from S×S output pixels
// where S = H_out/H_in = W_out/W_in (integer scale factor).
//
// Dispatch: (ceil(W_in/16), ceil(H_in/16), B*C)

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

void StoreNchw(uint buf_idx, uint b, uint c, uint h, uint w, uint C, uint H, uint W, float value) {
    uint idx = b * (C * H * W) + c * (H * W) + h * W + w;
    OaStore(heap[buf_idx], idx, value);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint x_in = gid.x * 16 + lid.x;
    uint y_in = gid.y * 16 + lid.y;
    uint bc = gid.z;

    if (x_in >= pc.w_in || y_in >= pc.h_in) return;

    uint b = bc / pc.channels;
    uint c = bc % pc.channels;
    if (b >= pc.batch_size) return;

    uint scale_y = pc.h_out / pc.h_in;
    uint scale_x = pc.w_out / pc.w_in;

    float sum = 0.0f;
    for (uint dy = 0; dy < scale_y; ++dy) {
        uint y_out = y_in * scale_y + dy;
        if (y_out >= pc.h_out) continue;
        for (uint dx = 0; dx < scale_x; ++dx) {
            uint x_out = x_in * scale_x + dx;
            if (x_out >= pc.w_out) continue;
            sum += LoadNchw(pc.in_idx, b, c, y_out, x_out, pc.channels, pc.h_out, pc.w_out);
        }
    }
    StoreNchw(pc.out_idx, b, c, y_in, x_in, pc.channels, pc.h_in, pc.w_in, sum);
}
