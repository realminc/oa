// Muon Nesterov momentum for 2D matrix step — writes lookahead update, updates momentum in place.
// Matches OaMuonRef::MatrixStep phase 1. Stream.cpp prepends buffer indices.

import Storage;

struct PushConstants {
    uint grad_idx;
    uint momentum_idx;
    uint update_idx;
    uint count;
    float beta;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= pc.count) return;

    float g = OaLoad(heap[pc.grad_idx], idx);
    float m = OaLoad(heap[pc.momentum_idx], idx);

    float m_new = pc.beta * m + (1.0f - pc.beta) * g;
    float update = (1.0f - pc.beta) * g + pc.beta * m_new;
    OaStore(heap[pc.momentum_idx], idx, m_new);
    OaStore(heap[pc.update_idx], idx, update);
}