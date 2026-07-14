// DequantizeQ4_K — Unpack Q4_K blocks to FP32
//
// Q4_K format: 256 elements per block, 4-bit quantization with hierarchical scales
// Block structure (144 bytes):
//   - d (FP16): super-block scale for quantized scales
//   - dmin (FP16): super-block scale for quantized mins
//   - scales[12]: 8 sub-block scales (6-bit each, packed)
//   - qs[128]: 256 × 4-bit quants (2 per byte)
//
// Dequantization formula: weight = (d * scale) * quant - (dmin * min)
//
// Dispatch: (numBlocks, 1, 1) where numBlocks = numElements / 256
// Each thread handles one 256-element block.

import Storage;

struct PushConstants {
    uint input_idx;   // Q4_K blocks buffer
    uint output_idx;  // FP32 output buffer
    uint num_blocks;  // Number of 256-element blocks
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

static const uint QK_K = 256;
static const uint K_SCALE_SIZE = 12;

// Helper: Extract 6-bit scale/min from packed scales array
// Q4_K packs 8 scales (6-bit each) into 12 bytes:
//   scales[0..3]: low 6 bits of scales 0-3
//   scales[4..7]: low 6 bits of mins 0-3
//   scales[8..11]: upper 2 bits of scales/mins 4-7 packed
void get_scale_min_k4(uint j, uint8_t scales[K_SCALE_SIZE], out uint8_t sc, out uint8_t m) {
    if (j < 4) {
        sc = scales[j] & 0x3F;
        m = scales[j + 4] & 0x3F;
    } else {
        sc = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
    }
   }
   
   [shader("compute")]
   [numthreads(256, 1, 1)]
   void main(uint3 tid : SV_DispatchThreadID) {
    uint blockIdx = tid.x;
    if (blockIdx >= pc.num_blocks) return;
    
    RWByteAddressBuffer input = heap[pc.input_idx];
    RWByteAddressBuffer output = heap[pc.output_idx];
    
    // Q4_K block layout (144 bytes):
    // [0..1]: d (FP16)
    // [2..3]: dmin (FP16)
    // [4..15]: scales[12]
    // [16..143]: qs[128]
    uint blockOffset = blockIdx * 144;
    
    // Load super-block scales
    uint16_t d_u16 = input.Load<uint16_t>(blockOffset + 0);
    uint16_t dmin_u16 = input.Load<uint16_t>(blockOffset + 2);
    float d = f16tof32(d_u16);
    float dmin = f16tof32(dmin_u16);
    
    // Load packed scales
    uint8_t scales[K_SCALE_SIZE];
    for (uint i = 0; i < K_SCALE_SIZE; ++i) {
        scales[i] = input.Load<uint8_t>(blockOffset + 4 + i);
    }
    
    // Dequantize 256 elements (8 sub-blocks of 32 elements each)
    uint qsOffset = blockOffset + 16;
    uint outOffset = blockIdx * QK_K * 4;  // FP32 = 4 bytes per element
    
    uint is = 0;  // sub-block index
    for (uint j = 0; j < QK_K; j += 64) {
        // Process 2 sub-blocks (64 elements) per iteration
        
        // Sub-block 0
        uint8_t sc0, m0;
        get_scale_min_k4(is + 0, scales, sc0, m0);
        float d1 = d * sc0;
        float m1 = dmin * m0;
        
        // Sub-block 1
        uint8_t sc1, m1_u8;
        get_scale_min_k4(is + 1, scales, sc1, m1_u8);
        float d2 = d * sc1;
        float m2 = dmin * m1_u8;
        
        // Dequantize 32 elements from low nibbles
        for (uint l = 0; l < 32; ++l) {
            uint8_t packed = input.Load<uint8_t>(qsOffset + l);
            uint8_t q = packed & 0x0F;
            float val = d1 * q - m1;
            output.Store(outOffset + (j + l) * 4, val);
        }
        
        // Dequantize 32 elements from high nibbles
        for (uint l = 0; l < 32; ++l) {
            uint8_t packed = input.Load<uint8_t>(qsOffset + l);
            uint8_t q = packed >> 4;
            float val = d2 * q - m2;
            output.Store(outOffset + (j + 32 + l) * 4, val);
        }
        
        qsOffset += 32;
        is += 2;
    }
}