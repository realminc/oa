// ============================================================================
// DequantizeInt8.slang — INT8 to FP32 dequantization
// ============================================================================
// Dequantizes INT8 tensor back to FP32:
//   output = quantized * scale
//
// Push constants:
//   [0..3]   = input buffer index (INT8)
//   [4..7]   = output buffer index (FP32)
//   [8..11]  = scale buffer index (FP32, single value)
//   [12..15] = element count
//
// Dispatch: (N + 255) / 256 workgroups of 256 threads
// ============================================================================

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[[vk::push_constant]] cbuffer PushConstants {
    uint input_idx;
    uint output_idx;
    uint scale_idx;
    uint count;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= count) return;

    // Load scale factor
    float scale = asfloat(heap[scale_idx].Load(0));
    
    // Load INT8 input
    int8_t quantized = heap[input_idx].Load<int8_t>(idx);
    
    // Dequantize: quantized * scale
    float val = float(quantized) * scale;
    
    // Store FP32 output
    heap[output_idx].Store(idx * 4, asuint(val));
}