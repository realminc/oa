// ============================================================================
// ComputeScaleQ4_0.slang — Compute per-block scale for Q4_0 quantization
// ============================================================================
// Computes scale = max(abs(block)) / 7 for each 32-element block
// Uses wave ops for efficient max reduction
//
// Push constants:
//   [0..3]   = input buffer index (FP32)
//   [4..7]   = scale buffer index (FP32, one per block)
//   [8..11]  = element count
//
// Dispatch: (N / 32 + 255) / 256 workgroups of 256 threads
// ============================================================================

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[[vk::push_constant]] cbuffer PushConstants {
    uint input_idx;
    uint scale_idx;
    uint count;
};

static const uint BLOCK_SIZE = 32;

groupshared float shared_max[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_GroupThreadID) {
    uint global_idx = tid.x;
    uint block_idx = global_idx / BLOCK_SIZE;
    uint lane = global_idx % BLOCK_SIZE;
    
    // Find max absolute value in this block
    float block_max = 0.0;
    for (uint i = lane; i < BLOCK_SIZE; i += 256) {
        uint idx = block_idx * BLOCK_SIZE + i;
        if (idx < count) {
            float val = asfloat(heap[input_idx].Load(idx * 4));
            block_max = max(block_max, abs(val));
        }
    }
    
    // Wave-level max reduction (no barrier needed)
    block_max = WaveActiveMax(block_max);
    
    // Store wave result for cross-wave reduction
    if (WaveGetLaneIndex() == 0) {
        shared_max[tid.x / WaveGetLaneCount()] = block_max;
    }
    GroupMemoryBarrierWithGroupSync();
    
    // Cross-wave reduction (only lane 0 of each wave stored)
    if (tid.x < 8) {  // 256 threads / 32 lanes = 8 wave leaders
        float wave_max = shared_max[tid.x];
        for (uint s = 4; s > 0; s >>= 1) {
            if (tid.x < s) wave_max = max(wave_max, shared_max[tid.x + s]);
        }
        if (tid.x == 0) {
            // Compute scale: max / 7 (Q4 range is -7 to 7)
            float scale = wave_max / 7.0;
            if (scale < 1e-6) scale = 1.0;  // Avoid division by zero
            heap[scale_idx].Store(block_idx * 4, asuint(scale));
        }
    }
}
