// ClipGradNormReduce — per-tensor sum-of-squares reduction for gradient norm clipping.
//
// Dispatch with N workgroups (one per grad tensor, N <= 16).
// Each workgroup accumulates sum(g*g) for its tensor and writes the partial
// result to partials[workgroup_id]. A subsequent ClipGradNormScale dispatch
// reads all partials, computes the global norm, and scales the grads in-place.
//
// Buffer binding order (Stream.cpp prepends indices in this order):
//   buf[0] = params    — uint[17]: [n_tensors, count0..count15]
//   buf[1] = partials  — float[16] output
//   buf[2..N+1] = grads
//
// Push constants: only the prepended buffer indices (no host payload).
// Total push = (2 + N) * 4 bytes, well within 128 bytes for N <= 16.

import Storage;

struct PushConstants {
    uint params_idx;       // prepended: buf[0]
    uint partials_idx;     // prepended: buf[1]
    uint grad_idx[16];     // prepended: buf[2..17]
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

groupshared float g_partial[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID, uint3 gid : SV_GroupThreadID, uint3 wid : SV_GroupID) {
    uint n_tensors = heap[pc.params_idx].Load(0);
    uint tensor    = wid.x;
    if (tensor >= n_tensors) return;

    uint lid = gid.x;
    uint n   = heap[pc.params_idx].Load((tensor + 1) * 4);
    uint gi  = pc.grad_idx[tensor];

    float acc = 0.0f;
    for (uint i = lid; i < n; i += 256) {
        float v = OaLoadF32(heap[gi], i);
        acc += v * v;
    }

    g_partial[lid] = acc;
    GroupMemoryBarrierWithGroupSync();

    for (uint s = 128; s > 0; s >>= 1) {
        if (lid < s) g_partial[lid] += g_partial[lid + s];
        GroupMemoryBarrierWithGroupSync();
    }

    if (lid == 0) {
        OaStoreF32(heap[pc.partials_idx], tensor, g_partial[0]);
    }
}
