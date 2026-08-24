// DrawRectOutline.slang - alpha-blended rectangle border on the compose image.

struct PushConstants {
    uint dst_idx;
    int dst_x;
    int dst_y;
    uint dst_w;
    uint dst_h;
    uint thickness;
    uint rgba;
    int clip_x;
    int clip_y;
    uint clip_w;
    uint clip_h;
    float corner_radius;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(1, 0)]] RWTexture2D<float4> images[];

float roundedRectDistance(float2 local, float2 halfSize, float radius) {
    float2 q = abs(local) - (halfSize - radius);
    return length(max(q, float2(0.0, 0.0)))
        + min(max(q.x, q.y), 0.0) - radius;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint x = gid.x * 8 + lid.x;
    uint y = gid.y * 8 + lid.y;
    if (x >= pc.dst_w || y >= pc.dst_h) return;

    uint thickness = max(pc.thickness, 1);
    float coverage = 1.0;
    float radius = min(
        max(pc.corner_radius, 0.0),
        0.5 * float(min(pc.dst_w, pc.dst_h)));
    if (radius <= 0.0) {
        bool border = x < thickness || y < thickness
            || x >= pc.dst_w - min(thickness, pc.dst_w)
            || y >= pc.dst_h - min(thickness, pc.dst_h);
        if (!border) return;
    } else {
        float2 halfSize = 0.5 * float2(pc.dst_w, pc.dst_h);
        float2 local = float2(x, y) + 0.5 - halfSize;
        float outerCoverage = saturate(
            0.5 - roundedRectDistance(local, halfSize, radius));
        float inset = min(float(thickness), min(halfSize.x, halfSize.y));
        float2 innerHalfSize = halfSize - inset;
        float innerCoverage = 0.0;
        if (innerHalfSize.x > 0.0 && innerHalfSize.y > 0.0) {
            float innerRadius = max(0.0, radius - inset);
            innerCoverage = saturate(
                0.5 - roundedRectDistance(local, innerHalfSize, innerRadius));
        }
        coverage = saturate(outerCoverage - innerCoverage);
        if (coverage <= 0.0) return;
    }

    int dstX = pc.dst_x + int(x);
    int dstY = pc.dst_y + int(y);
    if (dstX < pc.clip_x || dstY < pc.clip_y
        || uint(dstX - pc.clip_x) >= pc.clip_w
        || uint(dstY - pc.clip_y) >= pc.clip_h) return;
    uint imageWidth, imageHeight;
    images[pc.dst_idx].GetDimensions(imageWidth, imageHeight);
    if (dstX < 0 || dstY < 0
        || dstX >= int(imageWidth) || dstY >= int(imageHeight)) return;

    float4 src = float4(
        float((pc.rgba >> 24) & 0xff) / 255.0,
        float((pc.rgba >> 16) & 0xff) / 255.0,
        float((pc.rgba >> 8) & 0xff) / 255.0,
        float(pc.rgba & 0xff) / 255.0);
    src.a *= coverage;
    uint2 dst_coord = uint2(uint(dstX), uint(dstY));
    float4 dst = images[pc.dst_idx][dst_coord];
    images[pc.dst_idx][dst_coord] = float4(
        src.rgb * src.a + dst.rgb * (1.0 - src.a),
        src.a + dst.a * (1.0 - src.a));
}
