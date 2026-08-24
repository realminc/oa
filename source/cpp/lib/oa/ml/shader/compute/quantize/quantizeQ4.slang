// OA's native symmetric Q4 plane stores 32 values per block, one Float32 scale
// per block, and 16 payload bytes. One thread owns one aligned 32-bit payload
// word so no two invocations race on a packed byte.

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[[vk::push_constant]] cbuffer PushConstants {
    uint input_idx;
    uint output_idx;
    uint scale_idx;
    uint count;
};

static const uint BLOCK_SIZE = 32;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint word_idx = tid.x;
    uint block_count = count / BLOCK_SIZE + (count % BLOCK_SIZE != 0 ? 1 : 0);
    uint word_count = block_count * 4;
    if (word_idx >= word_count) return;

    uint block_idx = word_idx / 4;
    uint block_lane = (word_idx % 4) * 8;
    float scale = asfloat(heap[scale_idx].Load(block_idx * 4));
    bool scale_valid = isfinite(scale) && scale > 0.0;

    uint packed = 0;
    for (uint i = 0; i < 8; ++i) {
        uint global_idx = block_idx * BLOCK_SIZE + block_lane + i;
        float value = global_idx < count
            ? asfloat(heap[input_idx].Load(global_idx * 4))
            : 0.0;
        if (!isfinite(value)) value = 0.0;
        int quantized = scale_valid
            ? clamp(int(round(value / scale)), -7, 7)
            : 0;
        uint q4 = uint(quantized + 7);
        packed |= q4 << (i * 4);
    }

    heap[output_idx].Store(word_idx * 4, packed);
}
