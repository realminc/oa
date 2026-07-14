// ============================================================================
// QuantizeInt8.slang — Per-tensor symmetric INT8 quantization
// ============================================================================
// Quantizes FP32 tensor to INT8 using symmetric quantization:
//   scale = max(abs(tensor)) / 127
//   quantized = round(input / scale)
//
// Push constants:
//   [0..3]   = input buffer index (FP32)
//   [4..7]   = output buffer index (INT8)
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
    
    // Load FP32 input
    float val = asfloat(heap[input_idx].Load(idx * 4));
    
    // Quantize: round(val / scale), clamped to [-127, 127]
    float quantized_f = val / scale;
    int quantized = int(round(quantized_f));
    quantized = clamp(quantized, -127, 127);
    
    // Store as INT8 (using byte store)
    heap[output_idx].Store<int8_t>(idx, int8_t(quantized));
}