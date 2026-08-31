// Anti-aliased filled screen-space triangle.

struct PushConstants {
    uint dst_idx;
    float x0;
    float y0;
    float x1;
    float y1;
    float x2;
    float y2;
    uint rgba;
    uint bounds_x;
    uint bounds_y;
    uint bounds_w;
    uint bounds_h;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(1, 0)]] RWTexture2D<float4> images[];

float edge(float2 begin, float2 end, float2 point) {
    float2 segment = end - begin;
    float2 offset = point - begin;
    return segment.x * offset.y - segment.y * offset.x;
}

float signedEdgeDistance(
    float2 begin,
    float2 end,
    float2 point,
    float winding)
{
    float2 segment = end - begin;
    return winding * edge(begin, end, point)
        / max(length(segment), 1.0e-8);
}

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

    float2 p0 = float2(pc.x0, pc.y0);
    float2 p1 = float2(pc.x1, pc.y1);
    float2 p2 = float2(pc.x2, pc.y2);
    float area = edge(p0, p1, p2);
    if (abs(area) <= 1.0e-8) return;
    float winding = area < 0.0 ? -1.0 : 1.0;
    float2 pixel = float2(pc.bounds_x + tid.x, pc.bounds_y + tid.y) + 0.5;
    float distanceToEdge = min(
        signedEdgeDistance(p0, p1, pixel, winding),
        min(
            signedEdgeDistance(p1, p2, pixel, winding),
            signedEdgeDistance(p2, p0, pixel, winding)));
    float coverage = saturate(distanceToEdge + 0.5);
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
