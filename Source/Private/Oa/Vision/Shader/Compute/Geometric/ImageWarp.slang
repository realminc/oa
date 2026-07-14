// Remap, inverse affine warp and inverse perspective warp for NCHW images.

import Image;
import Storage;

struct PushConstants {
    uint in_idx;
    uint map_idx;
    uint out_idx;
    uint batch_size;
    uint channels;
    uint in_height;
    uint in_width;
    uint out_height;
    uint out_width;
    uint operation; // 0 remap, 1 affine, 2 perspective
    uint interpolation;
    uint border_mode;
    uint map_batch;
    float border_value;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

float Sample(uint b, uint c, float sy, float sx) {
    if (pc.interpolation == 0) {
        return OaLoadNchwBorder(heap[pc.in_idx], b, c,
            int(floor(sy + 0.5f)), int(floor(sx + 0.5f)),
            pc.channels, pc.in_height, pc.in_width, pc.border_mode, pc.border_value);
    }
    int y0 = int(floor(sy));
    int x0 = int(floor(sx));
    float fy = sy - float(y0);
    float fx = sx - float(x0);
    float v00 = OaLoadNchwBorder(heap[pc.in_idx], b, c, y0, x0,
        pc.channels, pc.in_height, pc.in_width, pc.border_mode, pc.border_value);
    float v01 = OaLoadNchwBorder(heap[pc.in_idx], b, c, y0, x0 + 1,
        pc.channels, pc.in_height, pc.in_width, pc.border_mode, pc.border_value);
    float v10 = OaLoadNchwBorder(heap[pc.in_idx], b, c, y0 + 1, x0,
        pc.channels, pc.in_height, pc.in_width, pc.border_mode, pc.border_value);
    float v11 = OaLoadNchwBorder(heap[pc.in_idx], b, c, y0 + 1, x0 + 1,
        pc.channels, pc.in_height, pc.in_width, pc.border_mode, pc.border_value);
    return lerp(lerp(v00, v01, fx), lerp(v10, v11, fx), fy);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint x = gid.x * 16 + lid.x;
    uint y = gid.y * 16 + lid.y;
    uint bc = gid.z;
    if (x >= pc.out_width || y >= pc.out_height ||
        bc >= pc.batch_size * pc.channels) return;
    uint b = bc / pc.channels;
    uint c = bc % pc.channels;
    float sx;
    float sy;
    if (pc.operation == 0) {
        uint mb = pc.map_batch == 1 ? 0 : b;
        uint plane = pc.out_height * pc.out_width;
        uint base = mb * (2 * plane) + y * pc.out_width + x;
        sx = OaLoad(heap[pc.map_idx], base);
        sy = OaLoad(heap[pc.map_idx], base + plane);
    } else if (pc.operation == 1) {
        float m0 = OaLoad(heap[pc.map_idx], 0);
        float m1 = OaLoad(heap[pc.map_idx], 1);
        float m2 = OaLoad(heap[pc.map_idx], 2);
        float m3 = OaLoad(heap[pc.map_idx], 3);
        float m4 = OaLoad(heap[pc.map_idx], 4);
        float m5 = OaLoad(heap[pc.map_idx], 5);
        sx = m0 * float(x) + m1 * float(y) + m2;
        sy = m3 * float(x) + m4 * float(y) + m5;
    } else {
        float m0 = OaLoad(heap[pc.map_idx], 0);
        float m1 = OaLoad(heap[pc.map_idx], 1);
        float m2 = OaLoad(heap[pc.map_idx], 2);
        float m3 = OaLoad(heap[pc.map_idx], 3);
        float m4 = OaLoad(heap[pc.map_idx], 4);
        float m5 = OaLoad(heap[pc.map_idx], 5);
        float m6 = OaLoad(heap[pc.map_idx], 6);
        float m7 = OaLoad(heap[pc.map_idx], 7);
        float m8 = OaLoad(heap[pc.map_idx], 8);
        float denominator = m6 * float(x) + m7 * float(y) + m8;
        if (abs(denominator) < 1.0e-20f) {
            sx = -1.0e20f;
            sy = -1.0e20f;
        } else {
            sx = (m0 * float(x) + m1 * float(y) + m2) / denominator;
            sy = (m3 * float(x) + m4 * float(y) + m5) / denominator;
        }
    }
    float value = Sample(b, c, sy, sx);
    OaStoreNchw(heap[pc.out_idx], b, c, y, x,
        pc.channels, pc.out_height, pc.out_width, value);
}
