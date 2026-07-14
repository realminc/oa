// NCHW grayscale and channel-reorder transforms.

import Storage;

struct PushConstants {
    uint in_idx;
    uint out_idx;
    uint batch_size;
    uint in_channels;
    uint out_channels;
    uint height;
    uint width;
    uint operation; // 0 = grayscale, 1 = reorder
    uint order0;
    uint order1;
    uint order2;
    uint order3;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

float Load(uint b, uint c, uint y, uint x) {
    uint i = b * (pc.in_channels * pc.height * pc.width) +
             c * (pc.height * pc.width) + y * pc.width + x;
    return OaLoad(heap[pc.in_idx], i);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint x = gid.x * 16 + lid.x;
    uint y = gid.y * 16 + lid.y;
    uint bc = gid.z;
    if (x >= pc.width || y >= pc.height || bc >= pc.batch_size * pc.out_channels) return;

    uint b = bc / pc.out_channels;
    uint c = bc % pc.out_channels;
    float result;
    if (pc.operation == 0) {
        result = 0.2126f * Load(b, 0, y, x) +
                 0.7152f * Load(b, 1, y, x) +
                 0.0722f * Load(b, 2, y, x);
    } else {
        uint order[4] = {pc.order0, pc.order1, pc.order2, pc.order3};
        result = Load(b, order[c], y, x);
    }

    uint out_i = b * (pc.out_channels * pc.height * pc.width) +
                 c * (pc.height * pc.width) + y * pc.width + x;
    OaStore(heap[pc.out_idx], out_i, result);
}
