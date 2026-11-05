// Donor: oa/ui/shader/compute/drawGlyphs.slang.
// One workgroup per coverage/SDF glyph instance.
struct PushConstants {
    uint glyph_idx; uint first; uint atlas_idx; uint count;
    uint batch; uint batch_count; uint dst_idx;
    int dst_x; int dst_y; uint dst_w; uint dst_h;
    int clip_x; int clip_y; uint clip_w; uint clip_h;
    uint atlas_w; uint atlas_h; float px_range;
};
[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];
// Compose storage has a fixed RGBA8_UNORM ABI; no formatless image access.
[[vk::image_format("rgba8")]]
[[vk::binding(1, 0)]] RWTexture2D<float4> images[];

float4 unpack_rgba(uint rgba) {
    return float4(float((rgba >> 24) & 255) / 255.0, float((rgba >> 16) & 255) / 255.0,
                  float((rgba >> 8) & 255) / 255.0, float(rgba & 255) / 255.0);
}
float load_atlas(int x, int y) {
    x = clamp(x, 0, int(pc.atlas_w) - 1); y = clamp(y, 0, int(pc.atlas_h) - 1);
    uint offset = uint(y) * pc.atlas_w + uint(x);
    uint word = heap[pc.atlas_idx].Load(offset & ~3u);
    return float((word >> ((offset & 3u) * 8u)) & 255u) / 255.0;
}
float sample_atlas(float2 position) {
    int2 p = int2(floor(position)); float2 f = frac(position);
    float a = lerp(load_atlas(p.x, p.y), load_atlas(p.x + 1, p.y), f.x);
    float b = lerp(load_atlas(p.x, p.y + 1), load_atlas(p.x + 1, p.y + 1), f.x);
    return lerp(a, b, f.y);
}
[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    if (gid.x >= pc.count) return;
    uint image_w, image_h; images[pc.dst_idx].GetDimensions(image_w, image_h);
    uint offset = (pc.first + gid.x) * 48u;
    float anchor_x = asfloat(heap[pc.glyph_idx].Load(offset));
    float anchor_y = asfloat(heap[pc.glyph_idx].Load(offset + 4));
    float offset_x = asfloat(heap[pc.glyph_idx].Load(offset + 8));
    float offset_y = asfloat(heap[pc.glyph_idx].Load(offset + 12));
    float width = asfloat(heap[pc.glyph_idx].Load(offset + 16));
    float height = asfloat(heap[pc.glyph_idx].Load(offset + 20));
    uint atlas_x = heap[pc.glyph_idx].Load(offset + 24);
    uint atlas_y = heap[pc.glyph_idx].Load(offset + 28);
    uint atlas_w = heap[pc.glyph_idx].Load(offset + 32);
    uint atlas_h = heap[pc.glyph_idx].Load(offset + 36);
    float4 color = unpack_rgba(heap[pc.glyph_idx].Load(offset + 40));
    uint flags = heap[pc.glyph_idx].Load(offset + 44);
    if ((flags >> 1) != pc.batch) return;
    bool bottom_to_top = (flags & 1) != 0;
    bool solid = atlas_w == 0 || atlas_h == 0;
    bool sdf = pc.px_range > 0.0;

    int x0 = int(floor(float(pc.dst_x) + anchor_x * float(pc.dst_w) + offset_x));
    int y0 = int(floor(float(pc.dst_y) + anchor_y * float(pc.dst_h) + offset_y));
    uint quad_w = uint(max(1.0, ceil(bottom_to_top ? height : width)));
    uint quad_h = uint(max(1.0, ceil(bottom_to_top ? width : height)));
    float smoothing = 0.0;
    if (!solid && sdf) {
        smoothing = clamp(
            0.5 * max(float(atlas_w) / width, float(atlas_h) / height) / pc.px_range,
            0.01, 0.25);
    }

    for (uint y = lid.y; y < quad_h; y += 8) {
        int dy = y0 + int(y);
        if (dy < pc.clip_y || uint(dy - pc.clip_y) >= pc.clip_h || dy < 0 || dy >= int(image_h)) continue;
        for (uint x = lid.x; x < quad_w; x += 8) {
            int dx = x0 + int(x);
            if (dx < pc.clip_x || uint(dx - pc.clip_x) >= pc.clip_w || dx < 0 || dx >= int(image_w)) continue;
            float alpha = color.a;
            if (!solid) {
                float2 source_position = bottom_to_top
                    ? float2(width - (float(y) + 0.5), float(x) + 0.5)
                    : float2(float(x) + 0.5, float(y) + 0.5);
                float2 uv = source_position / float2(width, height);
                float2 atlas_position = float2(float(atlas_x), float(atlas_y))
                    + uv * float2(float(atlas_w), float(atlas_h)) - 0.5;
                float coverage = sample_atlas(atlas_position);
                alpha *= sdf
                    ? smoothstep(0.5 - smoothing, 0.5 + smoothing, coverage)
                    : coverage;
            }
            if (alpha <= 0.001) continue;
            float4 dst = images[pc.dst_idx][uint2(dx, dy)];
            images[pc.dst_idx][uint2(dx, dy)] = float4(color.rgb * alpha + dst.rgb * (1.0 - alpha), alpha + dst.a * (1.0 - alpha));
        }
    }
}
