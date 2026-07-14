// Apply deterministic GPU-generated salt-and-pepper noise.

import Storage;

struct PushConstants {
    uint in_idx;
    uint random_idx;
    uint out_idx;
    uint num_elements;
    float probability;
    float salt_value;
    float pepper_value;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= pc.num_elements) return;
    float value = OaLoad(heap[pc.in_idx], i);
    float random = OaLoad(heap[pc.random_idx], i);
    float half_probability = pc.probability * 0.5f;
    if (random < half_probability) value = pc.pepper_value;
    else if (random >= 1.0f - half_probability) value = pc.salt_value;
    OaStore(heap[pc.out_idx], i, value);
}
