// RopeApplyBwd — backward pass for RopeApply.
//
// Rotation is an orthogonal transform so its inverse is its transpose:
//   dL/dx[i]       = dL/dx'[i] * cos - dL/dx'[i+half] * sin
//   dL/dx[i+half]  = dL/dx'[i] * sin + dL/dx'[i+half] * cos
//
// Same dispatch as RopeApply.

import Storage;
import Math;

struct PushConstants {
    uint grad_out_idx; // dL/dx' [T, num_heads * head_dim]
    uint grad_in_idx;  // dL/dx  [T, num_heads * head_dim]
    uint T;
    uint num_heads;
    uint head_dim;
    float theta_base;
    uint pos_offset;
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
    uint i    = rem % half_dim;

    uint pos  = t + pc.pos_offset;
    float freq = float(pos) * pow(pc.theta_base, -2.0f * float(i) / float(pc.head_dim));
    float c = cos(freq);
    float s = sin(freq);

    uint d0 = t * (pc.num_heads * pc.head_dim) + h * pc.head_dim + i;
    uint d1 = d0 + half_dim;

    float g0 = OaLoad(heap[pc.grad_out_idx], d0);
    float g1 = OaLoad(heap[pc.grad_out_idx], d1);

    OaStore(heap[pc.grad_in_idx], d0,  g0 * c + g1 * s);
    OaStore(heap[pc.grad_in_idx], d1, -g0 * s + g1 * c);
}
