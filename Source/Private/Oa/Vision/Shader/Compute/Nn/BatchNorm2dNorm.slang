// BatchNorm2dNorm.slang — Normalize + affine transform (Pass 2 of 2)
//
// y = gamma * (x - mean) / sqrt(var + eps) + beta  for input [N, C, H, W].
// Dispatch: (ceil(W/16), ceil(H/16), N*C)

import Storage;

struct PushConstants {
    uint in_idx;
    uint out_idx;
    uint mean_idx;
    uint var_idx;
    uint gamma_idx;
    uint beta_idx;
    uint batch_size;
    uint channels;
    uint height;
    uint width;
    float eps;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint x  = gid.x * 16 + lid.x;
    uint y  = gid.y * 16 + lid.y;
    uint bc = gid.z;

    if (x >= pc.width || y >= pc.height) return;

    uint b = bc / pc.channels;
    uint c = bc % pc.channels;
    if (b >= pc.batch_size) return;

    float mean  = OaLoad(heap[pc.mean_idx],  c);
    float var   = OaLoad(heap[pc.var_idx],   c);
    float gamma = OaLoad(heap[pc.gamma_idx], c);
    float beta  = OaLoad(heap[pc.beta_idx],  c);

    uint spatial = pc.height * pc.width;
    uint idx     = b * (pc.channels * spatial) + c * spatial + y * pc.width + x;
    float x_val  = OaLoad(heap[pc.in_idx], idx);

    float x_norm = (x_val - mean) * rsqrt(var + pc.eps);
    float y_val  = gamma * x_norm + beta;

    OaStore(heap[pc.out_idx], idx, y_val);
}
