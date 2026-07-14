// ClipGradNormScale — applies gradient norm clipping after ClipGradNormReduce.
//
// Dispatch with N workgroups (one per grad tensor, N <= 16).
// Reads partials[0..n_tensors-1], computes global norm, then scales every
// grad element by min(1, max_norm / norm). No-op if norm <= max_norm.
//
// Buffer binding order (Stream.cpp prepends indices in this order):
//   buf[0] = params    — uint[18]: [n_tensors, count0..count15, max_norm_bits]
//   buf[1] = partials  — float[16] (written by ClipGradNormReduce)
//   buf[2..N+1] = grads
//
// max_norm is carried in params[17] (float bit-pattern), NOT as a trailing push
// payload: the runtime packs only (2 + N) prepended buffer indices, so a payload
// after the fixed grad_idx[16] field would land at the wrong offset for N < 16
// (it read zero → scale 0 → every grad zeroed → training frozen). Reading it from
// the params buffer is independent of N.
//
// Push constants: prepended buffer indices only (no host payload).

import Storage;

struct PushConstants {
    uint  params_idx;      // prepended: buf[0]
    uint  partials_idx;    // prepended: buf[1]
    uint  grad_idx[16];    // prepended: buf[2..17]
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID, uint3 gid : SV_GroupThreadID, uint3 wid : SV_GroupID) {
    uint n_tensors = heap[pc.params_idx].Load(0);
    uint tensor    = wid.x;
    if (tensor >= n_tensors) return;

    // max_norm carried in params[17] as a float bit-pattern (see header).
    float max_norm = asfloat(heap[pc.params_idx].Load(17 * 4));

    // All threads read all partials and compute the same scale.
    float sum_sq = 0.0f;
    for (uint t = 0; t < n_tensors; t++) {
        sum_sq += OaLoadF32(heap[pc.partials_idx], t);
    }
    float norm  = sqrt(sum_sq);
    float scale = (norm > max_norm) ? (max_norm / norm) : 1.0f;

    if (scale >= 1.0f) return;

    uint lid = gid.x;
    uint n   = heap[pc.params_idx].Load((tensor + 1) * 4);
    uint gi  = pc.grad_idx[tensor];

    for (uint i = lid; i < n; i += 256) {
        OaStoreF32(heap[gi], i, OaLoadF32(heap[gi], i) * scale);
    }
}
