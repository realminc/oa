// BlitPlanar.slang — blit (and scale) planar multi-dtype image planes to a compose
// storage image.
//
// Each plane is a flat row-major array of W*H elements in its own dtype:
//   0 = U8    uint8   [0,255]   → normalized [0.0, 1.0]
//   1 = U16   uint16  [0,65535] → normalized [0.0, 1.0]
//   2 = F32   float32  passthrough
//   3 = BF16  bfloat16 decoded to float32, passthrough
//
// Channel layout: 1=gray (replicated to RGB), 3=RGB, 4=RGBA or RGB+depth.
// Missing alpha plane (channels < 4, or a_idx == OA_BINDLESS_INVALID) → a = 1.0.
// RGB channels are kept as-is (sRGB-encoded). Alpha/depth is linear (no gamma).
//
// dtypes_packed: bits[1:0]=R, bits[3:2]=G, bits[5:4]=B, bits[7:6]=A dtype.
//
// If dst_w != src_w or dst_h != src_h the source is nearest-neighbour scaled.
//
// Dispatch: (ceil(dst_w / 8), ceil(dst_h / 8), 1)

import Storage;

static const uint kInvalidSlot = 0xFFFFFFFF;

struct PushConstants {
    uint r_idx;        // bindless buffer index — R (or gray) plane
    uint g_idx;        // bindless buffer index — G plane  (kInvalidSlot if channels < 3)
    uint b_idx;        // bindless buffer index — B plane  (kInvalidSlot if channels < 3)
    uint a_idx;        // bindless buffer index — A/depth  (kInvalidSlot if channels < 4)
    uint dst_idx;      // bindless image index  — compose storage image
    uint src_w;
    uint src_h;
    uint dst_x;        // X offset in compose image (pixels)
    uint dst_y;        // Y offset in compose image (pixels)
    uint dtypes;       // packed per-channel dtype (2 bits each, LSB = R)
    uint channels;     // number of planes: 1, 2, 3, or 4
    uint dst_w;        // destination rect width  (pixels)
    uint dst_h;        // destination rect height (pixels)
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];
[[vk::binding(1, 0)]] RWTexture2D<float4> images[];

uint PlaneIndex(uint channel) {
    if (channel == 0) return pc.r_idx;
    if (channel == 1) return pc.g_idx;
    if (channel == 2) return pc.b_idx;
    return pc.a_idx;
}

uint DtypeOf(uint channel) {
    return (pc.dtypes >> (channel * 2)) & 0x3;
}

// Read one element from a typed plane buffer. Returns the raw [0,1] or float value.
float LoadPlane(uint buf_idx, uint pixel_idx, uint dtype) {
    if (dtype == 0) {
        // U8: pack ratio 4:1, need byte extraction
        uint byte_off = pixel_idx;
        uint word     = heap[buf_idx].Load(byte_off & ~3u);
        uint byte_val = (word >> ((byte_off & 3u) * 8)) & 0xFF;
        return float(byte_val) / 255.0f;
    }
    if (dtype == 1) {
        // U16: pack ratio 2:1, need halfword extraction
        uint byte_off  = pixel_idx * 2;
        uint word      = heap[buf_idx].Load(byte_off & ~3u);
        uint shift     = (byte_off & 2u) * 8;
        uint short_val = (word >> shift) & 0xFFFF;
        return float(short_val) / 65535.0f;
    }
    if (dtype == 2) {
        // F32: direct
        return asfloat(heap[buf_idx].Load(pixel_idx * 4));
    }
    // BF16: stored as little-endian uint16, shift into upper bits of uint32
    uint byte_off = pixel_idx * 2;
    uint word     = heap[buf_idx].Load(byte_off & ~3u);
    uint shift    = (byte_off & 2u) * 8;
    uint bits     = (word >> shift) & 0xFFFF;
    return asfloat(bits << 16);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint x = gid.x * 8 + lid.x;
    uint y = gid.y * 8 + lid.y;
    if (x >= pc.dst_w || y >= pc.dst_h) return;

    // Map output pixel to source pixel (nearest-neighbour).
    uint sx = min(x * pc.src_w / pc.dst_w, pc.src_w - 1);
    uint sy = min(y * pc.src_h / pc.dst_h, pc.src_h - 1);
    uint pixel = sy * pc.src_w + sx;

    float r, g, b, a;

    if (pc.channels == 1) {
        // Grayscale — replicate to RGB
        r = LoadPlane(pc.r_idx, pixel, DtypeOf(0));
        g = r;
        b = r;
        a = 1.0f;
    } else {
        r = LoadPlane(pc.r_idx, pixel, DtypeOf(0));
        g = LoadPlane(pc.g_idx, pixel, DtypeOf(1));
        b = LoadPlane(pc.b_idx, pixel, DtypeOf(2));
        if (pc.channels >= 4 && pc.a_idx != kInvalidSlot) {
            // Alpha / depth: linear, no sRGB decode
            a = LoadPlane(pc.a_idx, pixel, DtypeOf(3));
        } else {
            a = 1.0f;
        }
    }

    images[pc.dst_idx][uint2(pc.dst_x + x, pc.dst_y + y)] = float4(r, g, b, a);
}
