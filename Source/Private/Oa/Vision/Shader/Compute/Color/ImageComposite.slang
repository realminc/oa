// NCHW alpha blend, masked composite and rectangular erase.

import Storage;

struct PushConstants {
    uint a_idx;
    uint b_idx;
    uint mask_idx;
    uint out_idx;
    uint batch_size;
    uint channels;
    uint height;
    uint width;
    uint operation; // 0 = alpha blend, 1 = mask composite, 2 = erase
    uint mask_channels;
    uint rect_x;
    uint rect_y;
    uint rect_width;
    uint rect_height;
    float alpha;
    float value;
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
    uint i = b * (pc.channels * pc.height * pc.width) +
             c * (pc.height * pc.width) + y * pc.width + x;
    float a = OaLoad(heap[pc.a_idx], i);
    float result = a;
    if (pc.operation == 0) {
        float other = OaLoad(heap[pc.b_idx], i);
        result = a * (1.0f - pc.alpha) + other * pc.alpha;
    } else if (pc.operation == 1) {
        float other = OaLoad(heap[pc.b_idx], i);
        uint mc = pc.mask_channels == 1 ? 0 : c;
        uint mi = b * (pc.mask_channels * pc.height * pc.width) +
                  mc * (pc.height * pc.width) + y * pc.width + x;
        float mask = clamp(OaLoad(heap[pc.mask_idx], mi), 0.0f, 1.0f);
        result = a * (1.0f - mask) + other * mask;
    } else {
        bool inside = x >= pc.rect_x && x < pc.rect_x + pc.rect_width &&
                      y >= pc.rect_y && y < pc.rect_y + pc.rect_height;
        result = inside ? pc.value : a;
    }
    OaStore(heap[pc.out_idx], i, result);
}
