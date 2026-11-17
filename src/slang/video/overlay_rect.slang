// Rectangle overlay for native-YUV frames.
//
// One dispatch per box per plane component: inside the scaled rect the
// plane's fill code is written, otherwise the source byte is copied through.
// Later dispatches may bind the destination as their source to composite
// additional boxes; each invocation owns exactly one output element.

import attributes;
import storage;

struct PushConstants {
	uint src_index;
	uint dst_index;
	uint plane_width;
	uint plane_height;
	uint rect_x;
	uint rect_y;
	uint rect_w;
	uint rect_h;
	uint elem_stride;
	uint src_component;
	uint dst_component;
	uint fill_value;
	uint border_width;
};
[[vk::push_constant]] PushConstants push;

[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("overlay_rect")]
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

	uint src_offset = (y * push.plane_width + x) * push.elem_stride + push.src_component;
	uint value = load_u8(storage_buffers[push.src_index], src_offset);
	bool inside = x >= push.rect_x && x < push.rect_x + push.rect_w && y >= push.rect_y
		&& y < push.rect_y + push.rect_h;
	if (inside && push.border_width == 0u) {
		value = push.fill_value;
	} else if (inside && push.border_width != 0u) {
		bool ring = x < push.rect_x + push.border_width
			|| x + push.border_width >= push.rect_x + push.rect_w
			|| y < push.rect_y + push.border_width
			|| y + push.border_width >= push.rect_y + push.rect_h;
		if (ring) {
			value = push.fill_value;
		}
	}
	uint dst_offset = (y * push.plane_width + x) * push.elem_stride + push.dst_component;
	store_u8(storage_buffers[push.dst_index], dst_offset, value);
}
