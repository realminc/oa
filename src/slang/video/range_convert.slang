// Limited <-> full range conversion for native-YUV planes.
//
// Direction 0 expands limited to full; direction 1 packs full to limited.
// Component kind selects luma (219/224 headroom math) or chroma (224 offset
// math). Values round to nearest and saturate.

import attributes;
import storage;

struct PushConstants {
	uint src_index;
	uint dst_index;
	uint plane_width;
	uint plane_height;
	uint elem_stride;
	uint component;
	uint component_kind;
	uint direction;
};
[[vk::push_constant]] PushConstants push;

[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

uint convert(uint code, uint kind, uint direction) {
	if (direction == 2u) {
		return code;
	}
	float value = float(code);
	float converted;
	if (direction == 0u) {
		converted = kind == 0u ? (value - 16.0f) * 255.0f / 219.0f
			: (value - 128.0f) * 255.0f / 224.0f + 128.0f;
	} else {
		converted = kind == 0u ? value * 219.0f / 255.0f + 16.0f
			: (value - 128.0f) * 224.0f / 255.0f + 128.0f;
	}
	return uint(round(clamp(converted, 0.0f, 255.0f)));
}

[kernel_name("range_convert")]
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
	store_u8(storage_buffers[push.dst_index], offset, convert(code, push.component_kind, push.direction));
}
