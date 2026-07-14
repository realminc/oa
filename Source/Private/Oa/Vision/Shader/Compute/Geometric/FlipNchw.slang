// FlipNchw.slang — horizontal/vertical flip for NCHW tensors
//
// Dispatch: (ceil(W/16), ceil(H/16), B*C)

import Storage;

struct PushConstants {
    uint in_idx;
    uint out_idx;
    uint batch_size;
    uint channels;
    uint height;
    uint width;
    uint horizontal;
    uint vertical;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint x = gid.x * 16 + lid.x;
    uint y = gid.y * 16 + lid.y;
    uint bc = gid.z;
    if (x >= pc.width || y >= pc.height) return;

    uint b = bc / pc.channels;
    uint c = bc % pc.channels;
    if (b >= pc.batch_size) return;

    uint src_x = pc.horizontal != 0 ? (pc.width - 1 - x) : x;
    uint src_y = pc.vertical != 0 ? (pc.height - 1 - y) : y;
    uint src_idx = b * (pc.channels * pc.height * pc.width) + c * (pc.height * pc.width) + src_y * pc.width + src_x;
    uint dst_idx = b * (pc.channels * pc.height * pc.width) + c * (pc.height * pc.width) + y * pc.width + x;
    OaStore(heap[pc.out_idx], dst_idx, OaLoad(heap[pc.in_idx], src_idx));
}
