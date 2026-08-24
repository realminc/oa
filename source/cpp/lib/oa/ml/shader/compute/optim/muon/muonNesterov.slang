// Muon Nesterov momentum for 2D matrix step — writes lookahead update, updates momentum in place.
// The independent CPU oracle in testOptimMuon.cpp validates this phase.

import storage;

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

  float g = oaLoad(heap[pc.grad_idx], idx);
  float m = oaLoad(heap[pc.momentum_idx], idx);

  float m_new = pc.beta * m + (1.0f - pc.beta) * g;
  float update = (1.0f - pc.beta) * g + pc.beta * m_new;
  oaStore(heap[pc.momentum_idx], idx, m_new);
  oaStore(heap[pc.update_idx], idx, update);
}
