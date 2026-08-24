// All packed reads use aligned 32-bit addresses. The old byte-address load at
// an arbitrary byte offset was invalid for RWByteAddressBuffer and could read
// neighboring payload bytes differently across drivers.

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
    
    float scale = asfloat(heap[scale_idx].Load(block_idx * 4));
    uint block_word = lane / 8;
    uint packed = heap[input_idx].Load((block_idx * 4 + block_word) * 4);
    uint q4 = (packed >> ((lane % 8) * 4)) & 0x0F;
    int quantized = int(q4) - 7;
    float val = isfinite(scale) && scale > 0.0
        ? float(quantized) * scale
        : 0.0;
    heap[output_idx].Store(global_idx * 4, asuint(val));
}
