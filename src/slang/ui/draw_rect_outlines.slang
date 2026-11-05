// Batched normalized rectangle borders.
//
// Donor: oa/ui/shader/compute/drawRectOutlines.slang. One workgroup draws one
// record's border layers; each record is the packed 32-byte detection layout.

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
// Compose storage has a fixed RGBA8_UNORM ABI; no formatless image access.
[[vk::image_format("rgba8")]]
[[vk::binding(1, 0)]] RWTexture2D<float4> images[];

float4 unpack_rgba(uint rgba) {
	return float4(
		float((rgba >> 24) & 0xff) / 255.0,
		float((rgba >> 16) & 0xff) / 255.0,
		float((rgba >> 8) & 0xff) / 255.0,
		float(rgba & 0xff) / 255.0);
}

void store_pixel(int x, int y, float4 color) {
	if (x < pc.clip_x || y < pc.clip_y) {
		return;
	}
	if (uint(x - pc.clip_x) >= pc.clip_w || uint(y - pc.clip_y) >= pc.clip_h) {
		return;
	}
	uint image_width, image_height;
	images[pc.dst_idx].GetDimensions(image_width, image_height);
	if (x < 0 || y < 0 || x >= int(image_width) || y >= int(image_height)) {
		return;
	}
	images[pc.dst_idx][uint2(uint(x), uint(y))] = color;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
	uint rect_index = gid.x;
	if (rect_index >= pc.count) {
		return;
	}

	uint offset = rect_index * 32u;
	float center_x = asfloat(heap[pc.rect_idx].Load(offset));
	float center_y = asfloat(heap[pc.rect_idx].Load(offset + 4));
	float width = asfloat(heap[pc.rect_idx].Load(offset + 8));
	float height = asfloat(heap[pc.rect_idx].Load(offset + 12));
	uint instance_rgba = heap[pc.rect_idx].Load(offset + 24);

	int x0 = int(floor(float(pc.dst_x) + (center_x - width * 0.5) * float(pc.dst_w)));
	int y0 = int(floor(float(pc.dst_y) + (center_y - height * 0.5) * float(pc.dst_h)));
	int x1 = int(ceil(float(pc.dst_x) + (center_x + width * 0.5) * float(pc.dst_w)));
	int y1 = int(ceil(float(pc.dst_y) + (center_y + height * 0.5) * float(pc.dst_h)));

	bool beyond_right = x0 >= pc.clip_x && uint(x0 - pc.clip_x) >= pc.clip_w;
	bool beyond_bottom = y0 >= pc.clip_y && uint(y0 - pc.clip_y) >= pc.clip_h;
	if (x1 <= pc.clip_x || y1 <= pc.clip_y || beyond_right || beyond_bottom) {
		return;
	}
	int rect_width = x1 - x0;
	int rect_height = y1 - y0;
	if (rect_width <= 0 || rect_height <= 0) {
		return;
	}

	uint border = min(pc.thickness, uint((min(rect_width, rect_height) + 1) / 2));
	float4 color = unpack_rgba(instance_rgba != 0u ? instance_rgba : pc.rgba);
	for (uint layer = 0; layer < border; ++layer) {
		int left = x0 + int(layer);
		int top = y0 + int(layer);
		int right = x1 - 1 - int(layer);
		int bottom = y1 - 1 - int(layer);

		for (uint x = lid.x; x < uint(right - left + 1); x += 64u) {
			store_pixel(left + int(x), top, color);
			if (bottom != top) {
				store_pixel(left + int(x), bottom, color);
			}
		}
		for (uint y = lid.x + 1u; y + 1u < uint(bottom - top + 1); y += 64u) {
			store_pixel(left, top + int(y), color);
			if (right != left) {
				store_pixel(right, top + int(y), color);
			}
		}
	}
}
