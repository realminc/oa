// Muon-private two-input linear combination.
// out = scale_a * a + scale_b * b

import storage;

struct PushConstants {
    uint a_idx;
    uint b_idx;
    uint out_idx;
    uint count;
    float scale_a;
    float scale_b;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= pc.count) return;

    float a = oaLoad(heap[pc.a_idx], idx);
    float b = oaLoad(heap[pc.b_idx], idx);
    oaStore(heap[pc.out_idx], idx, pc.scale_a * a + pc.scale_b * b);
}
