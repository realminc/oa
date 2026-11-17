// UI raster planar blit — scale and pack planar multi-dtype image planes into
// the compose attachment.
//
// Raster port of blit_planar.slang. The scissor covers the clip-visible
// region. The compute path stored the packed pixel directly, so this fragment
// is paired with a replace (blend-disabled) pipeline; absent planes bind the
// R plane slot and are never read.
//
// Push layout (32 bytes):
//   offset  0 : uint2 src_size
//   offset  8 : int2  dst_origin
//   offset 16 : uint  dtypes — packed per-channel dtype (2 bits each, R first)
//   offset 20 : uint  channels — 1 gray, 2 gray+alpha, 3 RGB, 4 RGBA/depth
//   offset 24 : uint2 dst_size

struct PushConstants {
	uint src_w;
	uint src_h;
	int dst_x;
	int dst_y;
	uint dtypes;
	uint channels;
	uint dst_w;
	uint dst_h;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] ByteAddressBuffer heap[4];

uint dtype_of(uint channel) {
	return (pc.dtypes >> (channel * 2)) & 0x3;
}

// Read one element from a typed plane buffer. Returns the raw [0,1] or float value.
float load_plane(uint buffer_slot, uint pixel_idx, uint dtype) {
	ByteAddressBuffer plane = heap[buffer_slot];
	if (dtype == 0) {
		uint byte_off = pixel_idx;
		uint word = plane.Load(byte_off & ~3u);
		uint byte_val = (word >> ((byte_off & 3u) * 8)) & 0xFF;
		return float(byte_val) / 255.0;
	}
	if (dtype == 1) {
		uint byte_off = pixel_idx * 2;
		uint word = plane.Load(byte_off & ~3u);
		uint shift = (byte_off & 2u) * 8;
		uint short_val = (word >> shift) & 0xFFFF;
		return float(short_val) / 65535.0;
	}
	if (dtype == 2) {
		return asfloat(plane.Load(pixel_idx * 4));
	}
	uint byte_off = pixel_idx * 2;
	uint word = plane.Load(byte_off & ~3u);
	uint shift = (byte_off & 2u) * 8;
	uint bits = (word >> shift) & 0xFFFF;
	return asfloat(bits << 16);
}

[shader("fragment")]
float4 main(float4 position : SV_Position) : SV_Target0 {
	uint x = uint(int(position.x) - pc.dst_x);
	uint y = uint(int(position.y) - pc.dst_y);
	if (x >= pc.dst_w || y >= pc.dst_h || pc.src_w == 0 || pc.src_h == 0) {
		discard;
	}

	uint sx = min(x * pc.src_w / pc.dst_w, pc.src_w - 1);
	uint sy = min(y * pc.src_h / pc.dst_h, pc.src_h - 1);
	uint pixel = sy * pc.src_w + sx;

	float r;
	float g;
	float b;
	float a;

	if (pc.channels <= 2) {
		r = load_plane(0, pixel, dtype_of(0));
		g = r;
		b = r;
		a = pc.channels == 2 ? load_plane(1, pixel, dtype_of(1)) : 1.0;
	} else {
		r = load_plane(0, pixel, dtype_of(0));
		g = load_plane(1, pixel, dtype_of(1));
		b = load_plane(2, pixel, dtype_of(2));
		a = pc.channels >= 4 ? load_plane(3, pixel, dtype_of(3)) : 1.0;
	}
	return float4(r, g, b, a);
}
