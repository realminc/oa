// Portable fused Q8 MatMulNt for OA's explicit two-plane weight format.
// One 32-thread workgroup owns one output element and reduces over K without
// a subgroup-width assumption or an intermediate Float32 weight matrix.

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[[vk::push_constant]] cbuffer PushConstants {
    uint input_idx;
    uint payload_idx;
    uint scale_idx;
    uint output_idx;
    uint m;
    uint n;
    uint k;
};

static const uint BLOCK_SIZE = 32;
groupshared float partial[32];

[shader("compute")]
[numthreads(32, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID) {
    uint output_column = group.x;
    uint output_row = group.y;
    if (output_column >= n || output_row >= m) return;

    uint lane = thread.x;
    float accum = 0.0;
    for (uint input_column = lane; input_column < k; input_column += 32) {
        uint weight_index = output_column * k + input_column;
        uint block_index = weight_index / BLOCK_SIZE;
        uint block_lane = weight_index % BLOCK_SIZE;
        uint packed = heap[payload_idx].Load(
            (block_index * 8 + block_lane / 4) * 4);
        uint raw = (packed >> ((block_lane % 4) * 8)) & 0xFF;
        int quantized = raw >= 128 ? int(raw) - 256 : int(raw);
        float scale = asfloat(heap[scale_idx].Load(block_index * 4));
        float weight = isfinite(scale) && scale > 0.0
            ? float(quantized) * scale
            : 0.0;
        float input_value = asfloat(heap[input_idx].Load(
            (output_row * k + input_column) * 4));
        accum += input_value * weight;
    }

    partial[lane] = accum;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 16; stride > 0; stride >>= 1) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane == 0) {
        heap[output_idx].Store(
            (output_row * n + output_column) * 4, asuint(partial[0]));
    }
}
