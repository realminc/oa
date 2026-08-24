// Muon Vector/Scalar step — Nesterov momentum + decoupled weight decay (no NS5).
// The independent CPU oracle in testOptimMuon.cpp validates this path. Canonical
// language-model recipes normally keep one-dimensional parameters on AdamW.

import storage;

struct PushConstants {
    uint param_idx;
    uint grad_idx;
    uint momentum_idx;
    uint count;
    float lr;
    float beta;
    float weight_decay;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= pc.count) return;

    float p = oaLoad(heap[pc.param_idx], idx);
    float g = oaLoad(heap[pc.grad_idx], idx);
    float m = oaLoad(heap[pc.momentum_idx], idx);

    // Nesterov: m_new = beta*m + (1-beta)*g; update = (1-beta)*g + beta*m_new
    float m_new = pc.beta * m + (1.0f - pc.beta) * g;
    float update = (1.0f - pc.beta) * g + pc.beta * m_new;
    oaStore(heap[pc.momentum_idx], idx, m_new);

    // Keller/Moonshot decoupled WD: p = (1 - lr*wd)*p - lr*update
    float p_new = (1.0f - pc.lr * pc.weight_decay) * p - pc.lr * update;
    oaStore(heap[pc.param_idx], idx, p_new);
}
