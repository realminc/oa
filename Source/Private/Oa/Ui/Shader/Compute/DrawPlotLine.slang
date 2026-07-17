// One-dispatch line plot over a host-visible, bindless FP32 sample buffer.

import Storage;

struct PushConstants {
    uint values_idx;
    uint count;
    uint dst_idx;
    uint dst_x;
    uint dst_y;
    uint dst_w;
    uint dst_h;
    float y_min;
    float y_max;
    uint rgba;
    uint flags; // bit 0 = grid, bit 1 = fill
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

void blendPixel(uint2 coord, float4 src) {
    float4 dst = images[pc.dst_idx][coord];
    images[pc.dst_idx][coord] = float4(
        src.rgb * src.a + dst.rgb * (1.0 - src.a),
        src.a + dst.a * (1.0 - src.a));
}

float sampleAt(float position) {
    if (pc.count <= 1) return OaLoad(heap[pc.values_idx], 0);
    float index = clamp(position, 0.0, 1.0) * float(pc.count - 1);
    uint lo = uint(floor(index));
    uint hi = min(pc.count - 1, lo + 1);
    return lerp(
        OaLoad(heap[pc.values_idx], lo),
        OaLoad(heap[pc.values_idx], hi),
        index - float(lo));
}

int plotY(float value) {
    float normalized = saturate((value - pc.y_min) / (pc.y_max - pc.y_min));
    return int(round((1.0 - normalized) * float(pc.dst_h - 1)));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint x = tid.x;
    if (x >= pc.dst_w || pc.count == 0 || pc.dst_h == 0) return;
    uint imageWidth;
    uint imageHeight;
    images[pc.dst_idx].GetDimensions(imageWidth, imageHeight);
    if (pc.dst_x + x >= imageWidth || pc.dst_y >= imageHeight) return;
    uint visibleHeight = min(pc.dst_h, imageHeight - pc.dst_y);
    if (visibleHeight == 0) return;

    float position = pc.dst_w <= 1 ? 0.0 : float(x) / float(pc.dst_w - 1);
    float previousPosition = pc.dst_w <= 1 || x == 0
        ? position : float(x - 1) / float(pc.dst_w - 1);
    int y0 = min(plotY(sampleAt(previousPosition)), int(visibleHeight) - 1);
    int y1 = min(plotY(sampleAt(position)), int(visibleHeight) - 1);
    int lineMin = min(y0, y1);
    int lineMax = max(y0, y1);
    float4 color = unpackRgba(pc.rgba);

    if ((pc.flags & 1) != 0) {
        float4 grid = float4(1.0, 1.0, 1.0, 0.07);
        bool vertical = x == pc.dst_w / 4 || x == pc.dst_w / 2
            || x == (pc.dst_w * 3) / 4;
        for (uint y = 0; y < visibleHeight; ++y) {
            bool horizontal = y == pc.dst_h / 4 || y == pc.dst_h / 2
                || y == (pc.dst_h * 3) / 4;
            if (vertical || horizontal) {
                blendPixel(uint2(pc.dst_x + x, pc.dst_y + y), grid);
            }
        }
    }

    if ((pc.flags & 2) != 0) {
        float4 fill = float4(color.rgb, color.a * 0.16);
        for (int y = lineMax; y < int(visibleHeight); ++y) {
            blendPixel(uint2(pc.dst_x + x, pc.dst_y + uint(y)), fill);
        }
    }
    for (int y = lineMin; y <= lineMax; ++y) {
        blendPixel(uint2(pc.dst_x + x, pc.dst_y + uint(y)), color);
    }
}
