// Anti-aliased screen-space line segment.

struct PushConstants {
    uint dst_idx;
    float x0;
    float y0;
    float x1;
    float y1;
    float thickness;
    uint rgba;
    uint bounds_x;
    uint bounds_y;
    uint bounds_w;
    uint bounds_h;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(1, 0)]] RWTexture2D<float4> images[];

float4 unpackRgba(uint rgba) {
    return float4(
        float((rgba >> 24) & 0xff) / 255.0,
        float((rgba >> 16) & 0xff) / 255.0,
        float((rgba >> 8) & 0xff) / 255.0,
        float(rgba & 0xff) / 255.0);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= pc.bounds_w || tid.y >= pc.bounds_h) return;
    float2 pixel = float2(pc.bounds_x + tid.x, pc.bounds_y + tid.y) + 0.5;
    float2 begin = float2(pc.x0, pc.y0);
    float2 segment = float2(pc.x1, pc.y1) - begin;
    float lengthSquared = max(dot(segment, segment), 1.0e-8);
    float along = saturate(dot(pixel - begin, segment) / lengthSquared);
    float distanceToLine = length(pixel - (begin + along * segment));
    float coverage = 1.0 - smoothstep(
        max(0.0, pc.thickness * 0.5 - 1.0),
        pc.thickness * 0.5 + 1.0,
        distanceToLine);
    if (coverage <= 0.0) return;
    uint imageWidth;
    uint imageHeight;
    images[pc.dst_idx].GetDimensions(imageWidth, imageHeight);
    uint2 coordinate = uint2(pc.bounds_x + tid.x, pc.bounds_y + tid.y);
    if (coordinate.x >= imageWidth || coordinate.y >= imageHeight) return;
    float4 src = unpackRgba(pc.rgba);
    src.a *= coverage;
    float4 dst = images[pc.dst_idx][coordinate];
    images[pc.dst_idx][coordinate] = float4(
        src.rgb * src.a + dst.rgb * (1.0 - src.a),
        src.a + dst.a * (1.0 - src.a));
}
