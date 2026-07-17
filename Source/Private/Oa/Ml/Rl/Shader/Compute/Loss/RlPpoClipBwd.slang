// d(-mean(clipped PPO surrogate)) / d(new log probability).

import Storage;

struct PushConstants {
    uint new_log_probability_idx;
    uint old_log_probability_idx;
    uint advantage_idx;
    uint gradient_idx;
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
    bool clippingActive = (advantage >= 0.0 && ratio > 1.0 + pc.clip_epsilon)
        || (advantage < 0.0 && ratio < 1.0 - pc.clip_epsilon);
    float gradient = clippingActive
        ? 0.0
        : -(advantage * ratio) / float(pc.count);
    OaStore(heap[pc.gradient_idx], index, gradient);
}
