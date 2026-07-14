// NormalizeImage.slang — per-channel NCHW image normalization
//
// output = (input - mean[channel]) / std[channel]
// Dispatch: (ceil(W/16), ceil(H/16), B*C)

import Storage;

struct PushConstants {
    uint in_idx;
    uint out_idx;
    uint batch_size;
    uint channels;
    uint height;
    uint width;
    float mean0;
    float mean1;
    float mean2;
    float std0;
    float std1;
    float std2;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

float ChannelMean(uint c) {
    if (c == 0) return pc.mean0;
    if (c == 1) return pc.mean1;
    return pc.mean2;
}

float ChannelStd(uint c) {
    if (c == 0) return pc.std0;
    if (c == 1) return pc.std1;
    return pc.std2;
}

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

    uint idx = b * (pc.channels * pc.height * pc.width) + c * (pc.height * pc.width) + y * pc.width + x;
    float stdv = max(ChannelStd(c), 1.0e-12f);
    float value = (OaLoad(heap[pc.in_idx], idx) - ChannelMean(c)) / stdv;
    OaStore(heap[pc.out_idx], idx, value);
}
