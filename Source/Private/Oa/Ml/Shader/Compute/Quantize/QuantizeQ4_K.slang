// QuantizeQ4_K — Convert FP32 to Q4_K blocks
//
// Q4_K format: 256 elements per block, 4-bit quantization with hierarchical scales
// Block structure (144 bytes):
//   - d (FP16): super-block scale for quantized scales
//   - dmin (FP16): super-block scale for quantized mins
//   - scales[12]: 8 sub-block scales (6-bit each, packed)
//   - qs[128]: 256 × 4-bit quants (2 per byte)
//
// Quantization algorithm (from llama.cpp):
//   1. Divide 256 elements into 8 sub-blocks of 32 elements each
//   2. For each sub-block:
//      - Find min and max values
//      - Compute scale and min offset
//      - Quantize to 4-bit: q = round((x - min) / scale)
//   3. Quantize the 8 scales to 6-bit (hierarchical quantization)
//   4. Pack 4-bit quants into bytes (2 quants per byte)
//
// Dispatch: (numBlocks, 1, 1) where numBlocks = numElements / 256
// Each thread handles one 256-element block.

import Storage;

struct PushConstants {
    uint input_idx;   // FP32 input buffer
    uint output_idx;  // Q4_K blocks buffer
    uint num_blocks;  // Number of 256-element blocks
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

static const uint QK_K = 256;
static const uint K_SCALE_SIZE = 12;

// Helper: Quantize a float to 6-bit (0-63 range)
uint8_t quantize_to_6bit(float x, float scale) {
    int q = int(round(x / scale));
    return uint8_t(clamp(q, 0, 63));
   }
   
   // Helper: Find min and max in a sub-block
   void find_min_max(RWByteAddressBuffer input, uint offset, uint count, out float out_min, out float out_max) {
    float min_val = input.Load<float>(offset);
    float max_val = min_val;
    
    for (uint i = 1; i < count; ++i) {
        float val = input.Load<float>(offset + i * 4);
        min_val = min(min_val, val);
        max_val = max(max_val, val);
    }
    
    out_min = min_val;
    out_max = max_val;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint blockIdx = tid.x;
    if (blockIdx >= pc.num_blocks) return;
    
    RWByteAddressBuffer input = heap[pc.input_idx];
    RWByteAddressBuffer output = heap[pc.output_idx];
    
    uint inOffset = blockIdx * QK_K * 4;  // FP32 = 4 bytes per element
    uint blockOffset = blockIdx * 144;     // Q4_K block = 144 bytes
    
    // Step 1: Process 8 sub-blocks of 32 elements each
    float scales[8];
    float mins[8];
    uint8_t quants[QK_K];  // Temporary storage for 4-bit quants
    
    for (uint i = 0; i < 8; ++i) {
        uint subBlockOffset = inOffset + i * 32 * 4;
        
        // Find min and max in this sub-block
        float min_val, max_val;
        find_min_max(input, subBlockOffset, 32, min_val, max_val);
        
        // Compute scale and min
        float scale = (max_val - min_val) / 15.0f;  // 4-bit = 0-15 range
        if (scale < 1e-10f) scale = 1e-10f;  // Avoid division by zero
        
        scales[i] = scale;
        mins[i] = min_val;
        
        // Quantize 32 elements to 4-bit
        for (uint j = 0; j < 32; ++j) {
            float x = input.Load<float>(subBlockOffset + j * 4);
            float normalized = (x - min_val) / scale;
            int q = int(round(normalized));
            quants[i * 32 + j] = uint8_t(clamp(q, 0, 15));
        }
    }
    
    // Step 2: Find max scale and max min for hierarchical quantization
    float max_scale = 0.0f;
    float max_min = 0.0f;
    for (uint i = 0; i < 8; ++i) {
        max_scale = max(max_scale, scales[i]);
        max_min = max(max_min, abs(mins[i]));
    }
    
    // Compute super-block scales
    float d = max_scale / 63.0f;  // 6-bit scale quantization
    float dmin = max_min / 63.0f;  // 6-bit min quantization
    if (d < 1e-10f) d = 1e-10f;
    if (dmin < 1e-10f) dmin = 1e-10f;
    
    // Step 3: Quantize scales and mins to 6-bit
    uint8_t quantized_scales[8];
    uint8_t quantized_mins[8];
    
    for (uint i = 0; i < 8; ++i) {
        quantized_scales[i] = quantize_to_6bit(scales[i], d);
        quantized_mins[i] = quantize_to_6bit(abs(mins[i]), dmin);
    }
    
    // Step 4: Pack 6-bit scales into 12 bytes
    // Q4_K packing scheme (from llama.cpp):
    // scales[0..3]: low 6 bits of scales 0-3
    // scales[4..7]: low 6 bits of mins 0-3
    // scales[8..11]: upper 2 bits of scales/mins 4-7 packed
    uint8_t packed_scales[K_SCALE_SIZE];
    
    // Pack scales 0-3 (low 6 bits)
    for (uint i = 0; i < 4; ++i) {
        packed_scales[i] = quantized_scales[i] & 0x3F;
    }
    
    // Pack mins 0-3 (low 6 bits)
    for (uint i = 0; i < 4; ++i) {
        packed_scales[i + 4] = quantized_mins[i] & 0x3F;
    }
    
    // Pack upper 2 bits of scales/mins 4-7
    for (uint i = 0; i < 4; ++i) {
        uint8_t scale_upper = (quantized_scales[i + 4] >> 6) & 0x03;
        uint8_t min_upper = (quantized_mins[i + 4] >> 6) & 0x03;
        packed_scales[i + 8] = (quantized_scales[i + 4] & 0x0F) | ((quantized_mins[i + 4] & 0x0F) << 4);
    }
    
    // Step 5: Store super-block scales as FP16
    uint16_t d_u16 = uint16_t(f32tof16(d));
    uint16_t dmin_u16 = uint16_t(f32tof16(dmin));
    output.Store(blockOffset + 0, d_u16);
    output.Store(blockOffset + 2, dmin_u16);
    
    // Step 6: Store packed scales
    for (uint i = 0; i < K_SCALE_SIZE; ++i) {
        output.Store(blockOffset + 4 + i, packed_scales[i]);
    }
    
    // Step 7: Pack and store 4-bit quants (2 per byte)
    uint qsOffset = blockOffset + 16;
    for (uint i = 0; i < QK_K / 2; ++i) {
        uint8_t low = quants[i * 2];
        uint8_t high = quants[i * 2 + 1];
        uint8_t packed = (low & 0x0F) | ((high & 0x0F) << 4);
        output.Store(qsOffset + i, packed);
    }
}