// GemmInt8.slang — Weight-only INT8 quantized GEMM
// C[M, N] = A[M, K] * B[K, N]
// B is INT8 quantized weights, A and C are FP32.
// Dequantize-on-load: load INT8, convert to float, FP32 MAC.

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[[vk::push_constant]] cbuffer PushConstants {
    uint a_idx;
    uint b_idx;
    uint c_idx;
    uint scale_idx;
    uint M;
    uint N;
    uint K;
};

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
    uint row = gid.y * 16 + tid.y;
    uint col = gid.x * 16 + tid.x;

    if (row >= M || col >= N) return;

    float scale = asfloat(heap[scale_idx].Load(col * 4));
    float sum = 0.0F;

    for (uint k = 0; k < K; k++) {
        float a_val = asfloat(heap[a_idx].Load((row * K + k) * 4));
        uint b_byte_off = (k * N + col);
        uint b_raw = heap[b_idx].Load(b_byte_off);
        int b_val = (b_raw & 0x80u) != 0u ? int(b_raw | 0xFFFFFF00u) : int(b_raw);
        sum += a_val * float(b_val);
    }

    float result = sum * scale;
    heap[c_idx].Store((row * N + col) * 4, asuint(result));
}