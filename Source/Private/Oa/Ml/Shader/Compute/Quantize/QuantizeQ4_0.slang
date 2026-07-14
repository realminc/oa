// ============================================================================
// QuantizeQ4_0.slang — Per-block symmetric Q4 quantization (llama.cpp Q4_0)
// ============================================================================
// Quantizes FP32 tensor to Q4 using symmetric quantization:
//   Block size: 32 elements
//   scale = max(abs(block)) / 7
//   quantized = round(input / scale)
//   Pack 2 Q4 values per byte (nibbles)
//
// Push constants:
//   [0..3]   = input buffer index (FP32)
//   [4..7]   = output buffer index (Q4 packed)
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
    
    // Load FP32 input
    float val = asfloat(heap[input_idx].Load(global_idx * 4));
    
    // Quantize: round(val / scale), clamped to [-7, 7]
    float quantized_f = val / scale;
    int quantized = int(round(quantized_f));
    quantized = clamp(quantized, -7, 7);
    
    // Convert to unsigned 0-15 range (add 7 offset)
    uint q4 = uint(quantized + 7);
    
    // Pack: 2 Q4 values per byte
    // Lane 0 stores in lower nibble, lane 1 in upper nibble, etc.
    uint byte_idx = block_idx * (BLOCK_SIZE / 2) + lane / 2;
    uint packed = heap[output_idx].Load(byte_idx);
    
    if (lane % 2 == 0) {
        // Lower nibble
        packed = (packed & 0xF0) | q4;
    } else {
        // Upper nibble
        packed = (packed & 0x0F) | (q4 << 4);
    }
    
    heap[output_idx].Store(byte_idx, packed);
}
