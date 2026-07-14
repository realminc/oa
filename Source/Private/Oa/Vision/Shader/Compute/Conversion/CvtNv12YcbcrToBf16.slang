// CvtNv12YcbcrToBf16.slang - hardware YCbCr sampler -> BF16 NCHW tensor.
//
// The input image must be an NV12 VkImageView sampled through a sampler created
// with VkSamplerYcbcrConversion. The sampler returns RGB directly.

import Storage;

struct PushConstants {
    uint out_idx;
    uint nv12_idx;
    uint sampler_idx;
    uint width;
    uint height;
    uint coded_width;
    uint coded_height;
    uint normalize;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];
[[vk::binding(2, 0)]] Texture2D<float4> sampled_images[];
[[vk::binding(3, 0)]] SamplerState samplers[];

static const float3 IMAGENET_MEAN = float3(0.485f, 0.456f, 0.406f);
static const float3 IMAGENET_STD  = float3(0.229f, 0.224f, 0.225f);

void StoreBf16(uint index, float value) {
    uint bits = asuint(value);
    heap[pc.out_idx].Store<uint16_t>(index * 2u, uint16_t((bits >> 16u) & 0xFFFFu));
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint x = gid.x * 16 + lid.x;
    uint y = gid.y * 16 + lid.y;
    if (x >= pc.width || y >= pc.height) return;

    float2 uv = (float2(x, y) + 0.5f) / float2(pc.coded_width, pc.coded_height);
    float3 rgb = saturate(sampled_images[pc.nv12_idx].SampleLevel(samplers[pc.sampler_idx], uv, 0).rgb);

    if (pc.normalize != 0u) {
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD;
    }

    uint spatial = pc.width * pc.height;
    uint pixel = y * pc.width + x;
    StoreBf16(pixel, rgb.r);
    StoreBf16(spatial + pixel, rgb.g);
    StoreBf16(2 * spatial + pixel, rgb.b);
}
