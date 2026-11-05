// Donor: oa/source/cpp/lib/oa/core/shader/compute/flat/causalMaskBwd.slang
// Mechanical adaptation: OARS bindless names and FP32-only admitted storage.

import attributes;
import storage;

struct PushConstants {
	uint output_gradient_index;
	uint input_gradient_index;
	uint element_count;
	uint tq;
	uint tk;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("causal_mask_backward")]
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
	uint local = index % (push.tq * push.tk);
	uint q = local / push.tk;
	uint k = local % push.tk;
	float value = load_f32(storage_buffers[push.output_gradient_index], index);
	store_f32(
		storage_buffers[push.input_gradient_index],
		index,
		k > q ? 0.0f : value);
}
