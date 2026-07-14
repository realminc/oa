// RotateNchw.slang — rotate NCHW tensors by 90/180/270 degrees clockwise
//
// degrees: 90, 180, or 270
// Dispatch: (ceil(W_out/16), ceil(H_out/16), B*C)

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
    uint degrees;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint x = gid.x * 16 + lid.x;
    uint y = gid.y * 16 + lid.y;
    uint bc = gid.z;
    if (x >= pc.out_width || y >= pc.out_height) return;

    uint b = bc / pc.channels;
    uint c = bc % pc.channels;
    if (b >= pc.batch_size) return;

    uint src_x = x;
    uint src_y = y;
    if (pc.degrees == 90) {
        src_x = y;
        src_y = pc.in_height - 1 - x;
    } else if (pc.degrees == 180) {
        src_x = pc.in_width - 1 - x;
        src_y = pc.in_height - 1 - y;
    } else {
        src_x = pc.in_width - 1 - y;
        src_y = x;
    }

    uint src_idx = b * (pc.channels * pc.in_height * pc.in_width) + c * (pc.in_height * pc.in_width) + src_y * pc.in_width + src_x;
    uint dst_idx = b * (pc.channels * pc.out_height * pc.out_width) + c * (pc.out_height * pc.out_width) + y * pc.out_width + x;
    OaStore(heap[pc.out_idx], dst_idx, OaLoad(heap[pc.in_idx], src_idx));
}
