// Muon-private FP32 tiled GEMM with residual epilogue.
// out = alpha * (A @ B) + beta * residual.

import storage;

static const uint kTile = 16;

struct PushConstants {
    uint a_idx;
    uint b_idx;
    uint residual_idx;
    uint out_idx;
    uint M;
    uint N;
    uint K;
    float alpha;
    float beta;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

groupshared float shared_a[kTile * kTile];
groupshared float shared_b[kTile * kTile];

[shader("compute")]
[numthreads(kTile, kTile, 1)]
void main(
    uint3 groupId : SV_GroupID,
    uint3 groupThreadId : SV_GroupThreadID
) {
    uint localRow = groupThreadId.y;
    uint localColumn = groupThreadId.x;
    uint row = groupId.y * kTile + localRow;
    uint column = groupId.x * kTile + localColumn;
    float accumulator = 0.0f;

    for (uint tileK = 0; tileK < pc.K; tileK += kTile) {
        uint aK = tileK + localColumn;
        uint bK = tileK + localRow;
        shared_a[localRow * kTile + localColumn] = row < pc.M && aK < pc.K
            ? oaLoad(heap[pc.a_idx], row * pc.K + aK) : 0.0f;
        shared_b[localRow * kTile + localColumn] = bK < pc.K && column < pc.N
            ? oaLoad(heap[pc.b_idx], bK * pc.N + column) : 0.0f;
        GroupMemoryBarrierWithGroupSync();

        [[unroll]] for (uint k = 0; k < kTile; ++k) {
            accumulator += shared_a[localRow * kTile + k]
                * shared_b[k * kTile + localColumn];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (row < pc.M && column < pc.N) {
        uint index = row * pc.N + column;
        float residual = oaLoad(heap[pc.residual_idx], index);
        oaStore(heap[pc.out_idx], index,
            pc.alpha * accumulator + pc.beta * residual);
    }
}
