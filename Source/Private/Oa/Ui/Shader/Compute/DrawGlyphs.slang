// DrawGlyphs.slang - one workgroup per SDF glyph instance.

struct PushConstants {
    uint glyph_idx;
    uint atlas_idx;
    uint count;
    uint dst_idx;
    int dst_x;
    int dst_y;
    uint dst_w;
    uint dst_h;
    int clip_x;
    int clip_y;
    uint clip_w;
    uint clip_h;
    uint atlas_w;
    uint atlas_h;
    float px_range;
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

float loadAtlas(int x, int y) {
    x = clamp(x, 0, int(pc.atlas_w) - 1);
    y = clamp(y, 0, int(pc.atlas_h) - 1);
    uint byteOffset = uint(y) * pc.atlas_w + uint(x);
    uint word = heap[pc.atlas_idx].Load(byteOffset & ~3u);
    return float((word >> ((byteOffset & 3u) * 8u)) & 0xffu) / 255.0;
}

float sampleAtlas(float2 position) {
    int2 p0 = int2(floor(position));
    float2 f = frac(position);
    float a = lerp(loadAtlas(p0.x, p0.y), loadAtlas(p0.x + 1, p0.y), f.x);
    float b = lerp(loadAtlas(p0.x, p0.y + 1), loadAtlas(p0.x + 1, p0.y + 1), f.x);
    return lerp(a, b, f.y);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    if (gid.x >= pc.count) return;
    uint offset = gid.x * 48u;
    float anchorX = asfloat(heap[pc.glyph_idx].Load(offset));
    float anchorY = asfloat(heap[pc.glyph_idx].Load(offset + 4u));
    float offsetX = asfloat(heap[pc.glyph_idx].Load(offset + 8u));
    float offsetY = asfloat(heap[pc.glyph_idx].Load(offset + 12u));
    float width = asfloat(heap[pc.glyph_idx].Load(offset + 16u));
    float height = asfloat(heap[pc.glyph_idx].Load(offset + 20u));
    uint atlasX = heap[pc.glyph_idx].Load(offset + 24u);
    uint atlasY = heap[pc.glyph_idx].Load(offset + 28u);
    uint atlasW = heap[pc.glyph_idx].Load(offset + 32u);
    uint atlasH = heap[pc.glyph_idx].Load(offset + 36u);
    float4 color = unpackRgba(heap[pc.glyph_idx].Load(offset + 40u));

    int x0 = int(floor(float(pc.dst_x) + anchorX * float(pc.dst_w) + offsetX));
    int y0 = int(floor(float(pc.dst_y) + anchorY * float(pc.dst_h) + offsetY));
    uint quadW = uint(max(1.0, ceil(width)));
    uint quadH = uint(max(1.0, ceil(height)));
    float smoothing = clamp(
        0.5 * max(float(atlasW) / width, float(atlasH) / height) / pc.px_range,
        0.01, 0.25);

    for (uint y = lid.y; y < quadH; y += 8u) {
        int dstY = y0 + int(y);
        if (dstY < pc.clip_y || dstY >= pc.clip_y + int(pc.clip_h)) continue;
        for (uint x = lid.x; x < quadW; x += 8u) {
            int dstX = x0 + int(x);
            if (dstX < pc.clip_x || dstX >= pc.clip_x + int(pc.clip_w)) continue;
            float2 uv = (float2(float(x), float(y)) + 0.5)
                / float2(width, height);
            float2 atlasPosition = float2(float(atlasX), float(atlasY))
                + uv * float2(float(atlasW), float(atlasH)) - 0.5;
            float alpha = smoothstep(
                0.5 - smoothing, 0.5 + smoothing, sampleAtlas(atlasPosition));
            alpha *= color.a;
            if (alpha <= 0.001) continue;
            float4 dst = images[pc.dst_idx][uint2(uint(dstX), uint(dstY))];
            images[pc.dst_idx][uint2(uint(dstX), uint(dstY))] =
                float4(color.rgb * alpha + dst.rgb * (1.0 - alpha),
                    alpha + dst.a * (1.0 - alpha));
        }
    }
}
