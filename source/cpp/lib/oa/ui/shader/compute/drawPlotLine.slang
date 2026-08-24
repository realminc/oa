// One-dispatch line plot over a host-visible, bindless FP32 sample buffer.

import storage;

struct PushConstants {
    uint values_idx;
    uint count;
    uint dst_idx;
    int dst_x;
    int dst_y;
    uint dst_w;
    uint dst_h;
    float x_min;
    float x_max;
    float y_min;
    float y_max;
    uint rgba;
    uint flags; // bit 1 = fill, bit 2 = packed explicit X/Y
    uint antialias_samples; // admitted values: 1, 4, 8
    float line_width;
    int clip_x;
    int clip_y;
    uint clip_w;
    uint clip_h;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];
[[vk::binding(1, 0)]] RWTexture2D<float4> images[];

float4 unpackRgba(uint rgba) {
    return float4(
        float((rgba >> 24) & 0xff) / 255.0,
        float((rgba >> 16) & 0xff) / 255.0,
        float((rgba >> 8) & 0xff) / 255.0,
        float(rgba & 0xff) / 255.0);
}

void blendPixel(int2 coord, float4 src) {
    if (coord.x < pc.clip_x || coord.y < pc.clip_y
        || uint(coord.x - pc.clip_x) >= pc.clip_w
        || uint(coord.y - pc.clip_y) >= pc.clip_h) return;
    uint imageWidth, imageHeight;
    images[pc.dst_idx].GetDimensions(imageWidth, imageHeight);
    if (coord.x < 0 || coord.y < 0
        || coord.x >= int(imageWidth) || coord.y >= int(imageHeight)) return;
    uint2 pixel = uint2(uint(coord.x), uint(coord.y));
    float4 dst = images[pc.dst_idx][pixel];
    images[pc.dst_idx][pixel] = float4(
        src.rgb * src.a + dst.rgb * (1.0 - src.a),
        src.a + dst.a * (1.0 - src.a));
}

float sampleAt(float position) {
    if (pc.count <= 1) return oaLoad(heap[pc.values_idx], 0);
    float index = clamp(position, 0.0, 1.0) * float(pc.count - 1);
    uint lo = uint(floor(index));
    uint hi = min(pc.count - 1, lo + 1);
    return lerp(
        oaLoad(heap[pc.values_idx], lo),
        oaLoad(heap[pc.values_idx], hi),
        index - float(lo));
}

float plotY(float value) {
    float normalized = saturate((value - pc.y_min) / (pc.y_max - pc.y_min));
    return (1.0 - normalized) * float(pc.dst_h - 1);
}

float plotX(float value) {
    return saturate((value - pc.x_min) / (pc.x_max - pc.x_min))
        * float(pc.dst_w - 1);
}

float distanceToSegment(float2 point, float2 begin, float2 end) {
    float2 edge = end - begin;
    float lengthSquared = dot(edge, edge);
    if (lengthSquared <= 1.0e-8) return length(point - begin);
    float fraction = saturate(dot(point - begin, edge) / lengthSquared);
    return length(point - (begin + fraction * edge));
}

float2 sampleOffset(uint index, uint count) {
    if (count <= 1U) return float2(0.5, 0.5);
    if (count <= 4U) {
        return float2(
            (index & 1U) == 0U ? 0.25 : 0.75,
            (index & 2U) == 0U ? 0.25 : 0.75);
    }
    // Rotated 8-sample pattern avoids aligned stair-step repetition.
    switch (index & 7U) {
        case 0U: return float2(0.125, 0.375);
        case 1U: return float2(0.375, 0.875);
        case 2U: return float2(0.625, 0.125);
        case 3U: return float2(0.875, 0.625);
        case 4U: return float2(0.125, 0.875);
        case 5U: return float2(0.375, 0.125);
        case 6U: return float2(0.625, 0.625);
        default: return float2(0.875, 0.375);
    }
}

float lineCoverage(int2 pixel, float2 begin, float2 end) {
    uint count = pc.antialias_samples == 8U
        ? 8U : pc.antialias_samples == 1U ? 1U : 4U;
    float covered = 0.0;
    float halfWidth = max(0.25, pc.line_width * 0.5);
    for (uint sample = 0U; sample < 8U; ++sample) {
        if (sample >= count) break;
        float distance = distanceToSegment(
            float2(pixel) + sampleOffset(sample, count), begin, end);
        covered += 1.0 - smoothstep(
            max(0.0, halfWidth - 0.45), halfWidth + 0.45, distance);
    }
    return covered / float(count);
}

void drawSegmentColumn(uint x, float2 begin, float2 end, float4 color) {
    float radius = max(1.0, pc.line_width * 0.5 + 1.0);
    if (float(x) + 1.0 < min(begin.x, end.x) - radius
        || float(x) > max(begin.x, end.x) + radius) return;
    int minimumY = max(0, int(floor(min(begin.y, end.y) - radius)));
    int maximumY = min(int(pc.dst_h) - 1,
        int(ceil(max(begin.y, end.y) + radius)));
    for (int y = minimumY; y <= maximumY; ++y) {
        float coverage = lineCoverage(int2(int(x), y), begin, end);
        if (coverage <= 0.0) continue;
        blendPixel(
            int2(pc.dst_x + int(x), pc.dst_y + y),
            float4(color.rgb, color.a * coverage));
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint x = tid.x;
    if (x >= pc.dst_w || pc.count == 0 || pc.dst_h == 0) return;

    float4 color = unpackRgba(pc.rgba);

    if ((pc.flags & 4) != 0) {
        if (pc.x_max <= pc.x_min || pc.y_max <= pc.y_min) return;
        if (pc.count == 1) {
            float dataX = oaLoad(heap[pc.values_idx], 0);
            float dataY = oaLoad(heap[pc.values_idx], 1);
            if (isfinite(dataX) && isfinite(dataY)) {
                float2 point = float2(plotX(dataX), plotY(dataY));
                drawSegmentColumn(x, point, point, color);
            }
            return;
        }
        for (uint segment = 0; segment + 1 < pc.count; ++segment) {
            float dataX0 = oaLoad(heap[pc.values_idx], segment);
            float dataX1 = oaLoad(heap[pc.values_idx], segment + 1);
            float dataY0 = oaLoad(heap[pc.values_idx], pc.count + segment);
            float dataY1 = oaLoad(heap[pc.values_idx], pc.count + segment + 1);
            if (!isfinite(dataX0) || !isfinite(dataX1)
                || !isfinite(dataY0) || !isfinite(dataY1)) continue;
            float screenX0 = plotX(dataX0);
            float screenX1 = plotX(dataX1);
            float minimumX = min(screenX0, screenX1);
            float maximumX = max(screenX0, screenX1);
            if (abs(screenX1 - screenX0) < 0.5) {
                drawSegmentColumn(
                    x,
                    float2(screenX0, plotY(dataY0)),
                    float2(screenX1, plotY(dataY1)),
                    color);
                continue;
            }
            if (float(x) + 1.0 < minimumX - pc.line_width
                || float(x) > maximumX + pc.line_width) continue;
            drawSegmentColumn(
                x,
                float2(screenX0, plotY(dataY0)),
                float2(screenX1, plotY(dataY1)),
                color);
        }
        return;
    }

    float position = pc.dst_w <= 1 ? 0.0 : float(x) / float(pc.dst_w - 1);
    float previousPosition = pc.dst_w <= 1 || x == 0
        ? position : float(x - 1) / float(pc.dst_w - 1);
    float y0 = plotY(sampleAt(previousPosition));
    float y1 = plotY(sampleAt(position));

    if ((pc.flags & 2) != 0) {
        float4 fill = float4(color.rgb, color.a * 0.16);
        for (int y = int(ceil(max(y0, y1))); y < int(pc.dst_h); ++y) {
            blendPixel(int2(pc.dst_x + int(x), pc.dst_y + y), fill);
        }
    }
    drawSegmentColumn(
        x,
        float2(float(x == 0U ? 0U : x - 1U), y0),
        float2(float(x), y1),
        color);
}
