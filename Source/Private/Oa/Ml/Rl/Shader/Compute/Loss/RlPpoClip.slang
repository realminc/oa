// Per-sample negative clipped PPO surrogate. The C++ wrapper reduces by mean.

import Storage;

struct PushConstants {
    uint new_log_probability_idx;
    uint old_log_probability_idx;
    uint advantage_idx;
    uint output_idx;
    uint count;
    float clip_epsilon;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint index = tid.x;
    if (index >= pc.count) return;
    float ratio = exp(
        OaLoad(heap[pc.new_log_probability_idx], index)
        - OaLoad(heap[pc.old_log_probability_idx], index));
    float advantage = OaLoad(heap[pc.advantage_idx], index);
    float clippedRatio = clamp(
        ratio, 1.0 - pc.clip_epsilon, 1.0 + pc.clip_epsilon);
    float objective = min(ratio * advantage, clippedRatio * advantage);
    OaStore(heap[pc.output_idx], index, -objective);
}
