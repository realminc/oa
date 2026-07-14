// Asymmetric NCHW padding with shared border semantics.

import Image;
import Storage;

struct PushConstants {
    uint in_idx;
    uint out_idx;
    uint batch_size;
    uint channels;
    uint in_height;
    uint in_width;
    uint out_height;
    uint out_width;
    uint pad_top;
    uint pad_left;
    uint border_mode;
    float border_value;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

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
    float value = OaLoadNchwBorder(heap[pc.in_idx], b, c,
        int(y) - int(pc.pad_top), int(x) - int(pc.pad_left),
        pc.channels, pc.in_height, pc.in_width, pc.border_mode, pc.border_value);
    OaStoreNchw(heap[pc.out_idx], b, c, y, x,
        pc.channels, pc.out_height, pc.out_width, value);
}
