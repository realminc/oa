// DrawRect.slang - alpha-blended filled rectangle on the compose image.

struct PushConstants {
    uint dst_idx;
    uint dst_x;
    uint dst_y;
    uint dst_w;
    uint dst_h;
    uint rgba;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(1, 0)]] RWTexture2D<float4> images[];

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint x = gid.x * 8 + lid.x;
    uint y = gid.y * 8 + lid.y;
    if (x >= pc.dst_w || y >= pc.dst_h) return;

    float4 src = float4(
        float((pc.rgba >> 24) & 0xff) / 255.0,
        float((pc.rgba >> 16) & 0xff) / 255.0,
        float((pc.rgba >> 8) & 0xff) / 255.0,
        float(pc.rgba & 0xff) / 255.0);
    uint2 dstCoord = uint2(pc.dst_x + x, pc.dst_y + y);
    float4 dst = images[pc.dst_idx][dstCoord];
    images[pc.dst_idx][dstCoord] = float4(
        src.rgb * src.a + dst.rgb * (1.0 - src.a),
        src.a + dst.a * (1.0 - src.a));
}
