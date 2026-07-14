// RopeApply — Rotary Position Embedding (RoPE) forward pass.
//
// Applies rotary embeddings in-place to Q or K: x[T, D] → x'[T, D]
// where D = num_heads * head_dim and the rotation pairs are
// (d, d + head_dim/2) within each head — llama/Granite style.
//
// For each token t, head h, pair index i in [0, head_dim/2):
//   freq   = (t + pos_offset) / theta_base^(2i / head_dim)
//   x'[t, h*head_dim + i]               = x[...+i] * cos(freq) - x[...+i+half] * sin(freq)
//   x'[t, h*head_dim + i + head_dim/2]  = x[...+i] * sin(freq) + x[...+i+half] * cos(freq)
//
// Dispatch: ceil(T * num_heads * (head_dim/2) / 256) workgroups × 256 threads.
// Each thread handles one (token, head, pair) triple.

import Storage;
import Math;

struct PushConstants {
    uint x_idx;       // input  [T, num_heads * head_dim]
    uint out_idx;     // output [T, num_heads * head_dim]  (may alias x_idx for in-place)
    uint T;           // sequence length (tokens)
    uint num_heads;   // number of heads
    uint head_dim;    // dimension per head
    float theta_base; // RoPE base frequency (default 10000.0)
    uint pos_offset;  // starting position index (for KV-cache extension)
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
    uint half_dim = pc.head_dim / 2u;
    uint total = pc.T * pc.num_heads * half_dim;
    uint idx = gid.x * 256u + tid.x;
    if (idx >= total) return;

    uint t    = idx / (pc.num_heads * half_dim);
    uint rem  = idx % (pc.num_heads * half_dim);
    uint h    = rem / half_dim;
    uint i    = rem % half_dim;  // pair index within head

    uint pos  = t + pc.pos_offset;
    float freq = float(pos) * pow(pc.theta_base, -2.0f * float(i) / float(pc.head_dim));
    float c = cos(freq);
    float s = sin(freq);

    uint d0 = t * (pc.num_heads * pc.head_dim) + h * pc.head_dim + i;
    uint d1 = d0 + half_dim;

    float x0 = OaLoad(heap[pc.x_idx], d0);
    float x1 = OaLoad(heap[pc.x_idx], d1);

    OaStore(heap[pc.out_idx], d0, x0 * c - x1 * s);
    OaStore(heap[pc.out_idx], d1, x0 * s + x1 * c);
}
