// Signed bytes are extracted from aligned 32-bit payload loads. Invalid scale
// blocks produce deterministic zeros and never poison downstream Float32 work.

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
    if (global_idx >= count) return;

    uint block_idx = global_idx / BLOCK_SIZE;
    uint lane = global_idx % BLOCK_SIZE;
    uint packed = heap[input_idx].Load((block_idx * 8 + lane / 4) * 4);
    uint raw = (packed >> ((lane % 4) * 8)) & 0xFF;
    int quantized = raw >= 128 ? int(raw) - 256 : int(raw);
    float scale = asfloat(heap[scale_idx].Load(block_idx * 4));
    float value = isfinite(scale) && scale > 0.0
        ? float(quantized) * scale
        : 0.0;
    heap[output_idx].Store(global_idx * 4, asuint(value));
}
