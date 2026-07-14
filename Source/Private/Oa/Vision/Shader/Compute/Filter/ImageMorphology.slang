// Rectangular NCHW erosion/dilation with shared OA border semantics.

import Image;
import Storage;

struct PushConstants {
    uint in_idx;
    uint out_idx;
    uint batch_size;
    uint channels;
    uint height;
    uint width;
    uint kernel_height;
    uint kernel_width;
    uint operation; // 0 = erode/min, 1 = dilate/max
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
    int radius_y = int(pc.kernel_height / 2);
    int radius_x = int(pc.kernel_width / 2);
    float result = pc.operation == 0 ? 3.402823466e+38f : -3.402823466e+38f;

    for (uint ky = 0; ky < pc.kernel_height; ++ky) {
        int source_y = int(y) + int(ky) - radius_y;
        for (uint kx = 0; kx < pc.kernel_width; ++kx) {
            int source_x = int(x) + int(kx) - radius_x;
            float sample = OaLoadNchwBorder(
                heap[pc.in_idx], b, c, source_y, source_x,
                pc.channels, pc.height, pc.width, pc.border_mode, pc.border_value);
            result = pc.operation == 0 ? min(result, sample) : max(result, sample);
        }
    }

    OaStoreNchw(heap[pc.out_idx], b, c, y, x,
                pc.channels, pc.height, pc.width, result);
}
