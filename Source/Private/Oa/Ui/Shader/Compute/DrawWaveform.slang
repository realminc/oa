// Draw a precomputed [Bins, 2] min/max audio envelope as a scrub surface.

import Storage;

struct PushConstants {
    uint envelope_idx;
    uint bins;
    uint dst_idx;
    uint dst_x;
    uint dst_y;
    uint dst_w;
    uint dst_h;
    float fraction;
    uint played_rgba;
    uint remaining_rgba;
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

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint x = tid.x;
    if (x >= pc.dst_w || pc.bins == 0 || pc.dst_h == 0) return;
    uint bin = min(pc.bins - 1, x * pc.bins / pc.dst_w);
    float minimum = min(0.0, clamp(OaLoad(heap[pc.envelope_idx], bin * 2), -1.0, 1.0));
    float maximum = max(0.0, clamp(OaLoad(heap[pc.envelope_idx], bin * 2 + 1), -1.0, 1.0));

    float halfHeight = float(pc.dst_h - 1) * 0.5;
    int y0 = int(round(halfHeight - maximum * halfHeight));
    int y1 = int(round(halfHeight - minimum * halfHeight));
    y0 = clamp(y0, 0, int(pc.dst_h) - 1);
    y1 = clamp(y1, y0, int(pc.dst_h) - 1);

    float progress = (float(x) + 0.5) / float(pc.dst_w);
    float4 color = unpackRgba(progress <= pc.fraction
        ? pc.played_rgba
        : pc.remaining_rgba);
    for (int y = y0; y <= y1; ++y) {
        blendPixel(uint2(pc.dst_x + x, pc.dst_y + uint(y)), color);
    }
}
