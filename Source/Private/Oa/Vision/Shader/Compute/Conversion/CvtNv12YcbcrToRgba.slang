// CvtNv12YcbcrToRgba.slang - hardware YCbCr sampler -> RGBA storage image.

struct PushConstants {
    uint out_idx;
    uint nv12_idx;
    uint sampler_idx;
    uint width;
    uint height;
    uint coded_width;
    uint coded_height;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(1, 0)]] RWTexture2D<float4> storage_images[];
[[vk::binding(2, 0)]] Texture2D<float4> sampled_images[];
[[vk::binding(3, 0)]] SamplerState samplers[];

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint x = gid.x * 16 + lid.x;
    uint y = gid.y * 16 + lid.y;
    if (x >= pc.width || y >= pc.height) return;

    float2 uv = (float2(x, y) + 0.5f) / float2(pc.width, pc.height);
    float4 color = sampled_images[pc.nv12_idx].SampleLevel(samplers[pc.sampler_idx], uv, 0);
    storage_images[pc.out_idx][uint2(x, y)] = float4(color.rgb, 1.0f);
}
