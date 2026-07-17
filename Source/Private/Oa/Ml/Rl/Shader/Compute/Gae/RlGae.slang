// Generalized advantage estimation over [Time, Environments] FP32 rollouts.

import Storage;

struct PushConstants {
    uint reward_idx;
    uint value_idx;
    uint next_value_idx;
    uint terminated_idx;
    uint truncated_idx;
    uint advantage_idx;
    uint return_idx;
    uint time;
    uint environments;
    float gamma;
    float lambda;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint environment = tid.x;
    if (environment >= pc.environments) return;

    float nextAdvantage = 0.0;
    for (uint reverseTime = pc.time; reverseTime > 0; --reverseTime) {
        uint timeIndex = reverseTime - 1;
        uint index = timeIndex * pc.environments + environment;
        bool terminated = heap[pc.terminated_idx].Load<uint8_t>(index) != 0;
        bool truncated = heap[pc.truncated_idx].Load<uint8_t>(index) != 0;
        float bootstrapMask = terminated ? 0.0 : 1.0;
        float traceMask = (terminated || truncated) ? 0.0 : 1.0;
        float value = OaLoad(heap[pc.value_idx], index);
        float delta = OaLoad(heap[pc.reward_idx], index)
            + pc.gamma * bootstrapMask * OaLoad(heap[pc.next_value_idx], index)
            - value;
        float advantage = delta
            + pc.gamma * pc.lambda * traceMask * nextAdvantage;
        OaStore(heap[pc.advantage_idx], index, advantage);
        OaStore(heap[pc.return_idx], index, advantage + value);
        nextAdvantage = advantage;
    }
}
