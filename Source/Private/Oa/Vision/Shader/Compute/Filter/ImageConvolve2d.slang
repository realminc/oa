// Same-shape NCHW 2D cross-correlation with a shared odd kernel.

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
    uint kernel_height;
    uint kernel_width;
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
    float sum = 0.0f;

    for (uint ky = 0; ky < pc.kernel_height; ++ky) {
        int source_y = int(y) + int(ky) - radius_y;
        for (uint kx = 0; kx < pc.kernel_width; ++kx) {
            int source_x = int(x) + int(kx) - radius_x;
            float sample = OaLoadNchwBorder(
                heap[pc.in_idx], b, c, source_y, source_x,
                pc.channels, pc.height, pc.width, pc.border_mode, pc.border_value);
            float coefficient = OaLoad(heap[pc.kernel_idx], ky * pc.kernel_width + kx);
            sum += sample * coefficient;
        }
    }

    OaStoreNchw(heap[pc.out_idx], b, c, y, x,
                pc.channels, pc.height, pc.width, sum);
}
