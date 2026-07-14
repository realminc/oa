// BlitImageRgba.slang — blit a sampled RGBA image into the compose image.

struct PushConstants {
    uint src_idx;
    uint dst_idx;
    uint src_w;
    uint src_h;
    uint dst_x;
    uint dst_y;
    uint dst_w;
    uint dst_h;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(1, 0)]] RWTexture2D<float4> images[];
[[vk::binding(2, 0)]] Texture2D<float4> sampled_images[];

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint x = gid.x * 8 + lid.x;
    uint y = gid.y * 8 + lid.y;
    if (x >= pc.dst_w || y >= pc.dst_h) return;

    uint sx = min(x * pc.src_w / pc.dst_w, pc.src_w - 1);
    uint sy = min(y * pc.src_h / pc.dst_h, pc.src_h - 1);
    float4 color = sampled_images[pc.src_idx].Load(int3(int(sx), int(sy), 0));
    images[pc.dst_idx][uint2(pc.dst_x + x, pc.dst_y + y)] = color;
}
