// ============================================================================
// DequantizeQ4_0.slang — Per-block symmetric Q4 dequantization (llama.cpp Q4_0)
// ============================================================================
// Dequantizes Q4 tensor to FP32 using symmetric quantization:
//   Block size: 32 elements
//   Unpack 2 Q4 values per byte (nibbles)
//   dequantized = (q4 - 7) * scale
//
// Push constants:
//   [0..3]   = input buffer index (Q4 packed)
//   [4..7]   = output buffer index (FP32)
//   [8..11]  = scale buffer index (FP32, one per block)
//   [12..15] = element count
//
// Dispatch: (N / 32 + 255) / 256 workgroups of 256 threads
// ============================================================================

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[[vk::push_constant]] cbuffer PushConstants {
    uint input_idx;
    uint output_idx;
    uint scale_idx;
    uint count;
};

static const uint BLOCK_SIZE = 32;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint global_idx = tid.x;
    uint block_idx = global_idx / BLOCK_SIZE;
    uint lane = global_idx % BLOCK_SIZE;
    
    if (global_idx >= count) return;
    
    // Load scale for this block
    float scale = asfloat(heap[scale_idx].Load(block_idx * 4));
    
    // Unpack: 2 Q4 values per byte
    uint byte_idx = block_idx * (BLOCK_SIZE / 2) + lane / 2;
    uint packed = heap[input_idx].Load(byte_idx);
    
    uint q4;
    if (lane % 2 == 0) {
        // Lower nibble
        q4 = packed & 0x0F;
    } else {
        // Upper nibble
        q4 = (packed >> 4) & 0x0F;
    }
    
    // Convert from unsigned 0-15 to signed -7 to 7
    int quantized = int(q4) - 7;
    
    // Dequantize
    float val = float(quantized) * scale;
    
    // Store FP32 output
    heap[output_idx].Store(global_idx * 4, asuint(val));
}
