// Per-pixel luma mapping for native-YUV vision operations.
//
// Modes:
//   0 - binary threshold: value >= threshold -> high else low
//   1 - inverted binary threshold
//   2 - invert: 255 - value
// Each invocation owns one output element.

import attributes;
import storage;

struct PushConstants {
	uint src_index;
	uint dst_index;
	uint plane_width;
	uint plane_height;
	uint elem_stride;
	uint component;
	uint mode;
	uint threshold;
	uint low_value;
	uint high_value;
};
[[vk::push_constant]] PushConstants push;

[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("luma_map")]
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
	uint code = load_u8(storage_buffers[push.src_index], offset);
	uint value;
	if (push.mode == 0u) {
		value = code >= push.threshold ? push.high_value : push.low_value;
	} else if (push.mode == 1u) {
		value = code >= push.threshold ? push.low_value : push.high_value;
	} else if (push.mode == 2u) {
		value = 255u - code;
	} else {
		value = code;
	}
	store_u8(storage_buffers[push.dst_index], offset, value);
}
