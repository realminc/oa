// DrawRectOutlines.slang - batched normalized rectangle borders.

struct PushConstants {
    uint rect_idx;
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
    uint thickness;
    uint rgba;
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

void storePixel(int x, int y, float4 color) {
    if (x < pc.clip_x || y < pc.clip_y) return;
    if (x >= pc.clip_x + int(pc.clip_w) || y >= pc.clip_y + int(pc.clip_h)) return;
    images[pc.dst_idx][uint2(uint(x), uint(y))] = color;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint rectIndex = gid.x;
    if (rectIndex >= pc.count) return;

    uint offset = rectIndex * 32;
    float centerX = asfloat(heap[pc.rect_idx].Load(offset));
    float centerY = asfloat(heap[pc.rect_idx].Load(offset + 4));
    float width = asfloat(heap[pc.rect_idx].Load(offset + 8));
    float height = asfloat(heap[pc.rect_idx].Load(offset + 12));
    uint instanceRgba = heap[pc.rect_idx].Load(offset + 24);

    int x0 = int(floor(float(pc.dst_x) + (centerX - width * 0.5) * float(pc.dst_w)));
    int y0 = int(floor(float(pc.dst_y) + (centerY - height * 0.5) * float(pc.dst_h)));
    int x1 = int(ceil(float(pc.dst_x) + (centerX + width * 0.5) * float(pc.dst_w)));
    int y1 = int(ceil(float(pc.dst_y) + (centerY + height * 0.5) * float(pc.dst_h)));

    x0 = max(x0, pc.clip_x);
    y0 = max(y0, pc.clip_y);
    x1 = min(x1, pc.clip_x + int(pc.clip_w));
    y1 = min(y1, pc.clip_y + int(pc.clip_h));
    int rectWidth = x1 - x0;
    int rectHeight = y1 - y0;
    if (rectWidth <= 0 || rectHeight <= 0) return;

    uint border = min(pc.thickness, uint((min(rectWidth, rectHeight) + 1) / 2));
    float4 color = unpackRgba(instanceRgba != 0 ? instanceRgba : pc.rgba);
    for (uint layer = 0; layer < border; ++layer) {
        int left = x0 + int(layer);
        int top = y0 + int(layer);
        int right = x1 - 1 - int(layer);
        int bottom = y1 - 1 - int(layer);

        for (uint x = lid.x; x < uint(right - left + 1); x += 64) {
            storePixel(left + int(x), top, color);
            if (bottom != top) storePixel(left + int(x), bottom, color);
        }
        for (uint y = lid.x + 1; y + 1 < uint(bottom - top + 1); y += 64) {
            storePixel(left, top + int(y), color);
            if (right != left) storePixel(right, top + int(y), color);
        }
    }
}
