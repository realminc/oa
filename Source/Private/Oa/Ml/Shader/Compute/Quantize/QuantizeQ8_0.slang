// QuantizeQ8_0 — Convert FP32 to Q8_0 blocks
//
// Q8_0 format: 32 elements per block, 8-bit symmetric quantization
// Block structure (34 bytes):
//   - d (FP16): scale factor
//   - qs[32]: 32 × int8 quants
//
// Quantization algorithm:
//   1. Find max absolute value in block: amax = max(|x[i]|)
//   2. Compute scale: d = amax / 127
//   3. Quantize: q[i] = round(x[i] / d), clamped to [-127, 127]
//
// Dispatch: (numBlocks, 1, 1) where numBlocks = numElements / 32
// Each thread handles one 32-element block.

import Storage;

struct PushConstants {
    uint input_idx;   // FP32 input buffer
    uint output_idx;  // Q8_0 blocks buffer
    uint num_blocks;  // Number of 32-element blocks
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

static const uint QK8_0 = 32;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint blockIdx = tid.x;
    if (blockIdx >= pc.num_blocks) return;
    
    RWByteAddressBuffer input = heap[pc.input_idx];
    RWByteAddressBuffer output = heap[pc.output_idx];
    
    uint inOffset = blockIdx * QK8_0 * 4;  // FP32 = 4 bytes per element
    uint blockOffset = blockIdx * 34;       // Q8_0 block = 34 bytes
    
    // Step 1: Find max absolute value
    float amax = 0.0f;
    for (uint i = 0; i < QK8_0; ++i) {
        float x = input.Load<float>(inOffset + i * 4);
        amax = max(amax, abs(x));
    }
    
    // Step 2: Compute scale
    // Use 127 instead of 128 to avoid overflow when quantizing exactly -amax
    float d = amax / 127.0f;
    float id = (d != 0.0f) ? (1.0f / d) : 0.0f;  // inverse scale
    
    // Store scale as FP16
    uint16_t d_u16 = uint16_t(f32tof16(d));
    output.Store(blockOffset + 0, d_u16);
    
    // Step 3: Quantize to int8
    for (uint i = 0; i < QK8_0; ++i) {
        float x = input.Load<float>(inOffset + i * 4);
        float q_f = x * id;
        int q_i = int(round(q_f));
        q_i = clamp(q_i, -127, 127);
        int8_t q = int8_t(q_i);
        output.Store(blockOffset + 2 + i, q);
    }
}