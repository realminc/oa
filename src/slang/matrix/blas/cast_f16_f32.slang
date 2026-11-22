// FP16 → FP32 storage cast. One invocation per element; the f16 load widens
// exactly to f32. Requires storageBuffer16BitAccess + shaderFloat16.

import storage;
import attributes;

struct PushConstants {
	uint input_index;
	uint output_index;
	uint element_count;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("cast_f16_f32")]
[domain("matrix")]
[variant("cast_f16_f32")]
[dtype("f16")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID) {
	uint element = dispatch_thread_id.x;
	if (element >= push.element_count) {
		return;
	}
	store_f32(
		storage_buffers[push.output_index],
		element,
		load_f16(storage_buffers[push.input_index], element)
	);
}
