// CvtNv12YcbcrToRgba.slang - hardware YCbCr sampler -> RGBA storage image.

struct PushConstants {
    uint out_idx;
    uint width;
    uint height;
    uint coded_width;
    uint coded_height;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(1, 0)]] RWTexture2D<float4> storage_images[];
// Vulkan YCbCr conversion requires one combined-image descriptor backed by an
// immutable sampler. Keep ordinary OA resources in global bindless set 0 and
// put the fixed YCbCr source in the decoder-owned set 1.
[[vk::binding(0, 1)]] Sampler2D<float4> ycbcr_image;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint x = gid.x * 16 + lid.x;
    uint y = gid.y * 16 + lid.y;
    if (x >= pc.width || y >= pc.height) return;

    float2 uv = (float2(x, y) + 0.5f) / float2(pc.coded_width, pc.coded_height);
    float4 color = ycbcr_image.SampleLevel(uv, 0);
    storage_images[pc.out_idx][uint2(x, y)] = float4(color.rgb, 1.0f);
}
