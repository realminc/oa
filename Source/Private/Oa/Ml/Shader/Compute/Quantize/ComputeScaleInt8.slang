// ComputeScaleInt8.slang — Compute INT8 quantization scale

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[[vk::push_constant]] cbuffer PushConstants {
    uint input_idx;
    uint output_idx;
    uint partial_idx;
    uint count;
    uint pass;
};

groupshared float shared_max[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uint3 dtid : SV_DispatchThreadID) {
    uint local_id = tid.x;
    uint global_id = dtid.x;

    float local_max = 0.0F;

    if (pass == 0) {
        if (global_id < count) {
            float val = asfloat(heap[input_idx].Load(global_id * 4));
            local_max = abs(val);
        }
    } else {
        if (global_id < count) {
            local_max = asfloat(heap[partial_idx].Load(global_id * 4));
        }
    }

    shared_max[local_id] = local_max;
    GroupMemoryBarrierWithGroupSync();

    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (local_id < stride) {
            shared_max[local_id] = max(shared_max[local_id], shared_max[local_id + stride]);
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (local_id == 0) {
        float result = shared_max[0];
        if (pass == 0) {
            heap[partial_idx].Store(gid.x * 4, asuint(result));
        } else {
            float scale = result / 127.0F;
            if (scale < 1e-8F) scale = 1e-8F;
            heap[output_idx].Store(0, asuint(scale));
        }
    }
}