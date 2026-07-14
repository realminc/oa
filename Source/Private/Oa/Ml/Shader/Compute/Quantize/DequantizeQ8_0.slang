// DequantizeQ8_0 — Unpack Q8_0 blocks to FP32
//
// Q8_0 format: 32 elements per block, 8-bit symmetric quantization
// Block structure (34 bytes):
//   - d (FP16): scale factor
//   - qs[32]: 32 × int8 quants
//
// Dequantization formula: weight = d * quant
//
// Dispatch: (numBlocks, 1, 1) where numBlocks = numElements / 32
// Each thread handles one 32-element block.

import Storage;

struct PushConstants {
    uint input_idx;   // Q8_0 blocks buffer
    uint output_idx;  // FP32 output buffer
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
    
    // Q8_0 block layout (34 bytes):
    // [0..1]: d (FP16)
    // [2..33]: qs[32] (int8)
    uint blockOffset = blockIdx * 34;
    
    // Load scale
    uint16_t d_u16 = input.Load<uint16_t>(blockOffset + 0);
    float d = f16tof32(d_u16);
    
    // Dequantize 32 elements
    uint outOffset = blockIdx * QK8_0 * 4;  // FP32 = 4 bytes per element
    
    for (uint i = 0; i < QK8_0; ++i) {
        int8_t q = input.Load<int8_t>(blockOffset + 2 + i);
        float val = d * float(q);
        output.Store(outOffset + i * 4, val);
    }
}