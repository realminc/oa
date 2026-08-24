// One portable thread owns one 32-value block. This intentionally avoids a
// subgroup-width contract: mobile, Intel, AMD and NVIDIA devices may expose
// different wave sizes, while a 32-value serial reduction is negligible next
// to the memory traffic of this preprocessing path.

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[[vk::push_constant]] cbuffer PushConstants {
    uint input_idx;
    uint scale_idx;
    uint count;
};

static const uint BLOCK_SIZE = 32;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint block_idx = tid.x;
    uint block_count = count / BLOCK_SIZE + (count % BLOCK_SIZE != 0 ? 1 : 0);
    if (block_idx >= block_count) return;

    float block_max = 0.0;
    for (uint i = 0; i < BLOCK_SIZE; ++i) {
        uint idx = block_idx * BLOCK_SIZE + i;
        if (idx < count) {
            float val = asfloat(heap[input_idx].Load(idx * 4));
            if (isfinite(val)) {
                block_max = max(block_max, abs(val));
            }
        }
    }

    float scale = block_max / 7.0;
    if (!(scale > 0.0) || !isfinite(scale)) scale = 1.0;
    heap[scale_idx].Store(block_idx * 4, asuint(scale));
}
