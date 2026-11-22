// Donor: oa/source/cpp/lib/oa/core/shader/compute/flat/equal.slang
// Mechanical adaptation: OARS bindless names and FP32-only admitted storage.

import attributes;
import storage;

struct PushConstants {
	uint input_index;
	uint output_index;
	uint element_count;
	float value;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("equal")]
[domain("matrix")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID) {
	uint index = dispatch_thread_id.x;
	if (index >= push.element_count) {
		return;
	}
	float input = load_f32(storage_buffers[push.input_index], index);
	store_f32(
		storage_buffers[push.output_index],
		index,
		input == push.value ? 1.0f : 0.0f);
}
