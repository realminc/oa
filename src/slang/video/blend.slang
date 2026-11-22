// Per-component plane blending for native-YUV compositing.
//
// Modes:
//   0 - stencil: where the mask byte is nonzero the fill code is written,
//       otherwise the base byte is copied.
//   1 - alpha: dst = front * a + back * (1 - a) with a = alpha / 255.
// Each invocation owns one output element; chroma planes run the same code
// at their own resolution (mask sampled at the chroma coordinate).

import attributes;
import storage;

struct PushConstants {
	uint back_index;
	uint front_index;
	uint dst_index;
	uint plane_width;
	uint plane_height;
	uint elem_stride;
	uint component;
	uint mode;
	uint alpha;
	uint fill_value;
	uint mask_width;
	uint mask_height;
	uint mask_scale;
};
[[vk::push_constant]] PushConstants push;

[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("blend")]
[domain("video")]
[variant("plane_u8")]
[dtype("u8")]
[status("experimental")]
[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint x = thread_id.x;
	uint y = thread_id.y;
	if (x >= push.plane_width || y >= push.plane_height) return;

	uint offset = (y * push.plane_width + x) * push.elem_stride + push.component;
	uint back = load_u8(storage_buffers[push.back_index], offset);
	uint value = back;
	if (push.mode == 0u) {
		uint mx = push.mask_scale == 0u ? x : x * push.mask_scale;
		uint my = push.mask_scale == 0u ? y : y * push.mask_scale;
		uint mask = 0u;
		if (mx < push.mask_width && my < push.mask_height) {
			mask = load_u8(storage_buffers[push.front_index], my * push.mask_width + mx);
		}
		if (mask != 0u) {
			value = push.fill_value;
		}
	} else {
		uint front = load_u8(storage_buffers[push.front_index], offset);
		value = (front * push.alpha + back * (255u - push.alpha) + 127u) / 255u;
	}
	store_u8(storage_buffers[push.dst_index], offset, value);
}
