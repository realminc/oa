// Muon Normalize
// Compute Frobenius norm and normalize: X_0 = m_t / ||m_t||_F

import Storage;

struct PushConstants {
    uint momentum_idx;
    uint x_idx;
    uint norm_idx;
    uint rows;
    uint cols;
    float eps;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

groupshared float shared_sum[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID, uint3 gid : SV_GroupThreadID) {
    uint size = pc.rows * pc.cols;
    uint local_id = gid.x;
    
    // Phase 1: Compute sum of squares (Frobenius norm squared)
    float local_sum = 0.0f;
    for (uint i = local_id; i < size; i += 256) {
        float val = OaLoad(heap[pc.momentum_idx], i);
        local_sum += val * val;
    }
    shared_sum[local_id] = local_sum;
    GroupMemoryBarrierWithGroupSync();
    
    // Reduction
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (local_id < stride) {
            shared_sum[local_id] += shared_sum[local_id + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    
    // Phase 2: Normalize
    float norm_sq = shared_sum[0];
    float norm = sqrt(norm_sq + pc.eps);
    
    if (local_id == 0) {
        OaStore(heap[pc.norm_idx], 0, norm);
    }
    
    // Normalize and write to X
    for (uint i = local_id; i < size; i += 256) {
        float val = OaLoad(heap[pc.momentum_idx], i);
        OaStore(heap[pc.x_idx], i, val / norm);
    }
}
