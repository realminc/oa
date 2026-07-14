// BatchNorm2dStats.slang — Per-channel mean + variance (Pass 1 of 2)
//
// Computes mean[C] and variance[C] over (N, H, W) for input [N, C, H, W].
// Dispatch: (C, 1, 1) — one workgroup per channel.

import Storage;

struct PushConstants {
    uint in_idx;
    uint mean_idx;
    uint var_idx;
    uint batch_size;
    uint channels;
    uint height;
    uint width;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

groupshared float sm_sum[256];
groupshared float sm_sq_sum[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint c   = gid.x;
    uint tid = lid.x;

    if (c >= pc.channels) return;

    uint spatial  = pc.height * pc.width;
    uint total    = pc.batch_size * spatial;

    float local_sum    = 0.0f;
    float local_sq_sum = 0.0f;

    for (uint i = tid; i < total; i += 256) {
        uint b   = i / spatial;
        uint hw  = i % spatial;
        uint h   = hw / pc.width;
        uint w   = hw % pc.width;
        uint idx = b * (pc.channels * spatial) + c * spatial + h * pc.width + w;
        float v  = OaLoad(heap[pc.in_idx], idx);
        local_sum    += v;
        local_sq_sum += v * v;
    }

    sm_sum[tid]    = local_sum;
    sm_sq_sum[tid] = local_sq_sum;
    GroupMemoryBarrierWithGroupSync();

    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sm_sum[tid]    += sm_sum[tid + stride];
            sm_sq_sum[tid] += sm_sq_sum[tid + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == 0) {
        float mean     = sm_sum[0]    / float(total);
        float sq_mean  = sm_sq_sum[0] / float(total);
        // E[x^2] - E[x]^2 can become slightly negative from FP32
        // cancellation for nearly constant, large-magnitude channels. A
        // variance is non-negative by definition; clamp before rsqrt in pass 2.
        float variance = max(sq_mean - mean * mean, 0.0f);
        OaStore(heap[pc.mean_idx], c, mean);
        OaStore(heap[pc.var_idx],  c, variance);
    }
}
