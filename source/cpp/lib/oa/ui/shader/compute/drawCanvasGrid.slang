// Unified decimal viewport/chart/canvas grid over one clipped rectangle.

struct PushConstants {
    uint dst_idx;
    int dst_x;
    int dst_y;
    uint dst_w;
    uint dst_h;
    float origin_x;
    float origin_y;
    float minor_spacing_x;
    float minor_spacing_y;
    uint major_every;
    uint super_major_every;
    float minor_thickness;
    float major_thickness;
    float super_major_thickness;
    float axis_thickness;
    uint background_top_rgba;
    uint background_bottom_rgba;
    uint minor_rgba;
    uint major_rgba;
    uint super_major_rgba;
    uint axis_rgba;
    uint flags; // bit 0 = gradient fill, bit 1 = axes, bit 2 = grid tiers
    int clip_x;
    int clip_y;
    uint clip_w;
    uint clip_h;
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

float4 blendOver(float4 dst, float4 src, float coverage) {
    src.a *= coverage;
    return float4(
        src.rgb * src.a + dst.rgb * (1.0 - src.a),
        src.a + dst.a * (1.0 - src.a));
}

float distanceToPeriodicLine(float value, float spacing) {
    return abs(frac(value / spacing + 0.5) - 0.5) * spacing;
}

float lineCoverage(float distanceToLine, float thickness) {
    return 1.0 - smoothstep(
        max(0.0, thickness * 0.5 - 0.75),
        thickness * 0.5 + 0.75,
        distanceToLine);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= pc.dst_w || tid.y >= pc.dst_h) return;
    int dstX = pc.dst_x + int(tid.x);
    int dstY = pc.dst_y + int(tid.y);
    if (dstX < pc.clip_x || dstY < pc.clip_y
        || uint(dstX - pc.clip_x) >= pc.clip_w
        || uint(dstY - pc.clip_y) >= pc.clip_h) return;

    uint imageWidth;
    uint imageHeight;
    images[pc.dst_idx].GetDimensions(imageWidth, imageHeight);
    if (dstX < 0 || dstY < 0
        || dstX >= int(imageWidth) || dstY >= int(imageHeight)) return;

    const uint2 coordinate = uint2(uint(dstX), uint(dstY));
    const float2 pixel = float2(float(dstX), float(dstY)) + 0.5;
    const float2 relative = pixel - float2(pc.origin_x, pc.origin_y);
    float4 color = images[pc.dst_idx][coordinate];
    if ((pc.flags & 1U) != 0U) {
        const float fraction = pc.dst_h <= 1U
            ? 0.0 : float(tid.y) / float(pc.dst_h - 1U);
        const float4 background = lerp(
            unpackRgba(pc.background_top_rgba),
            unpackRgba(pc.background_bottom_rgba),
            fraction);
        color = blendOver(color, background, 1.0);
    }

    if ((pc.flags & 4U) != 0U) {
        const float minorDistance = min(
            distanceToPeriodicLine(relative.x, pc.minor_spacing_x),
            distanceToPeriodicLine(relative.y, pc.minor_spacing_y));
        color = blendOver(color, unpackRgba(pc.minor_rgba),
            lineCoverage(minorDistance, pc.minor_thickness));

        const float2 majorSpacing = float2(
            pc.minor_spacing_x, pc.minor_spacing_y) * float(pc.major_every);
        const float majorDistance = min(
            distanceToPeriodicLine(relative.x, majorSpacing.x),
            distanceToPeriodicLine(relative.y, majorSpacing.y));
        color = blendOver(color, unpackRgba(pc.major_rgba),
            lineCoverage(majorDistance, pc.major_thickness));

        const float2 superMajorSpacing = float2(
            pc.minor_spacing_x, pc.minor_spacing_y)
            * float(pc.super_major_every);
        const float superMajorDistance = min(
            distanceToPeriodicLine(relative.x, superMajorSpacing.x),
            distanceToPeriodicLine(relative.y, superMajorSpacing.y));
        color = blendOver(color, unpackRgba(pc.super_major_rgba),
            lineCoverage(superMajorDistance, pc.super_major_thickness));
    }

    if ((pc.flags & 2U) != 0U) {
        const float axisDistance = min(abs(relative.x), abs(relative.y));
        color = blendOver(color, unpackRgba(pc.axis_rgba),
            lineCoverage(axisDistance, pc.axis_thickness));
    }
    images[pc.dst_idx][coordinate] = color;
}
