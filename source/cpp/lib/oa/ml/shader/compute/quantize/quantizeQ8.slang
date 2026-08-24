// OA's native symmetric Q8 plane stores 32 signed payload bytes and one
// separate Float32 scale per block. One invocation owns one aligned word.

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
    uint word_count = block_count * 8;
    if (word_idx >= word_count) return;

    uint block_idx = word_idx / 8;
    uint block_lane = (word_idx % 8) * 4;
    float scale = asfloat(heap[scale_idx].Load(block_idx * 4));
    bool scale_valid = isfinite(scale) && scale > 0.0;

    uint packed = 0;
    for (uint i = 0; i < 4; ++i) {
        uint global_idx = block_idx * BLOCK_SIZE + block_lane + i;
        float value = global_idx < count
            ? asfloat(heap[input_idx].Load(global_idx * 4))
            : 0.0;
        if (!isfinite(value)) value = 0.0;
        int quantized = scale_valid
            ? clamp(int(round(value / scale)), -127, 127)
            : 0;
        packed |= (uint(quantized) & 0xFF) << (i * 8);
    }

    heap[output_idx].Store(word_idx * 4, packed);
}
