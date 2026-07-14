// One pass of same-shape NCHW separable cross-correlation.

import Image;
import Storage;

struct PushConstants {
    uint in_idx;
    uint kernel_idx;
    uint out_idx;
    uint batch_size;
    uint channels;
    uint height;
    uint width;
    uint kernel_size;
    uint horizontal;
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
    if (x >= pc.width || y >= pc.height || bc >= pc.batch_size * pc.channels) return;

    uint b = bc / pc.channels;
    uint c = bc % pc.channels;
    int radius = int(pc.kernel_size / 2);
    float sum = 0.0f;

    for (uint k = 0; k < pc.kernel_size; ++k) {
        int offset = int(k) - radius;
        int source_y = int(y) + (pc.horizontal != 0 ? 0 : offset);
        int source_x = int(x) + (pc.horizontal != 0 ? offset : 0);
        float sample = OaLoadNchwBorder(
            heap[pc.in_idx], b, c, source_y, source_x,
            pc.channels, pc.height, pc.width, pc.border_mode, pc.border_value);
        sum += sample * OaLoad(heap[pc.kernel_idx], k);
    }

    OaStoreNchw(heap[pc.out_idx], b, c, y, x,
                pc.channels, pc.height, pc.width, sum);
}
