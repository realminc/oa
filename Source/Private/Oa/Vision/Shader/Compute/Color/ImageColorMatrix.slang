// Per-pixel 3x4 RGB color matrix for NCHW images.

import Storage;

struct PushConstants {
    uint in_idx;
    uint transform_idx;
    uint out_idx;
    uint batch_size;
    uint channels;
    uint height;
    uint width;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

float Load(uint b, uint c, uint y, uint x) {
    uint i = b * (pc.channels * pc.height * pc.width) +
             c * (pc.height * pc.width) + y * pc.width + x;
    return OaLoad(heap[pc.in_idx], i);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint x = gid.x * 16 + lid.x;
    uint y = gid.y * 16 + lid.y;
    uint bc = gid.z;
    if (x >= pc.width || y >= pc.height || bc >= pc.batch_size * pc.channels) return;
    uint b = bc / pc.channels;
    uint c = bc % pc.channels;
    float result;
    if (c < 3) {
        float r = Load(b, 0, y, x);
        float g = Load(b, 1, y, x);
        float blue = Load(b, 2, y, x);
        uint row = c * 4;
        result = OaLoad(heap[pc.transform_idx], row) * r +
                 OaLoad(heap[pc.transform_idx], row + 1) * g +
                 OaLoad(heap[pc.transform_idx], row + 2) * blue +
                 OaLoad(heap[pc.transform_idx], row + 3);
    } else {
        result = Load(b, c, y, x);
    }
    uint i = b * (pc.channels * pc.height * pc.width) +
             c * (pc.height * pc.width) + y * pc.width + x;
    OaStore(heap[pc.out_idx], i, result);
}
