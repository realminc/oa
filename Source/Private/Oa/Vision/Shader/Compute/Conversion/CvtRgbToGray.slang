// CvtRgbToGray.slang — RGB → Grayscale Conversion
// Standard luminance weights: Gray = 0.299*R + 0.587*G + 0.114*B
// Input: float[B, 3, H, W] (RGB)
// Output: float[B, 1, H, W] (grayscale)

import Storage;

struct PushConstants {
    uint in_idx;
    uint out_idx;
    uint batch_size;
    uint height;
    uint width;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint x = gid.x * 16 + lid.x;
    uint y = gid.y * 16 + lid.y;
    uint b = gid.z;
    if (x >= pc.width || y >= pc.height || b >= pc.batch_size) return;

    uint spatial = pc.height * pc.width;
    uint base = b * (3 * spatial) + y * pc.width + x;

    float r = OaLoad(heap[pc.in_idx], base);
    float g = OaLoad(heap[pc.in_idx], base + spatial);
    float bl = OaLoad(heap[pc.in_idx], base + 2 * spatial);

    float gray = 0.299f * r + 0.587f * g + 0.114f * bl;

    uint out_idx = b * spatial + y * pc.width + x;
    OaStore(heap[pc.out_idx], out_idx, gray);
}