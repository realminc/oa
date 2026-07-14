// Muon matrix apply — Moonshot scaled orthogonal update + decoupled weight decay.
// p = (1 - lr*wd)*p - lr*moonshot_scale*ortho
// moonshot_scale = 0.2 * sqrt(max(rows, cols)) (computed on host).

import Storage;

struct PushConstants {
    uint param_idx;
    uint ortho_idx;
    uint count;
    float lr;
    float weight_decay;
    float moonshot_scale;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= pc.count) return;

    float p = OaLoad(heap[pc.param_idx], idx);
    float ortho = OaLoad(heap[pc.ortho_idx], idx);

    float p_new = (1.0f - pc.lr * pc.weight_decay) * p - pc.lr * pc.moonshot_scale * ortho;
    OaStore(heap[pc.param_idx], idx, p_new);
}