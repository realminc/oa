// Per-component plane stencil driven by device-resident f32 scores.
//
// Where the score at the mask coordinate is at or above the threshold the
// fill code is written, otherwise the base byte is copied. Each invocation
// owns one output element; chroma planes run the same code at their own
// resolution (the score is sampled at the scaled luma coordinate).

import attributes;
import storage;

struct PushConstants {
	uint back_index;
	uint score_index;
	uint dst_index;
	uint plane_width;
	uint plane_height;
	uint elem_stride;
	uint component;
	float threshold;
	uint fill_value;
	uint mask_width;
	uint mask_height;
	uint mask_scale;
};
[[vk::push_constant]] PushConstants push;

[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("score_blend")]
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
	uint mx = push.mask_scale == 0u ? x : x * push.mask_scale;
	uint my = push.mask_scale == 0u ? y : y * push.mask_scale;
	float score = -1.0;
	if (mx < push.mask_width && my < push.mask_height) {
		score = load_f32(storage_buffers[push.score_index], my * push.mask_width + mx);
	}
	uint value = score >= push.threshold ? push.fill_value : back;
	store_u8(storage_buffers[push.dst_index], offset, value);
}
