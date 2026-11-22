// One portable thread owns one 32-value block. This intentionally avoids a
// subgroup-width contract: mobile, Intel, AMD and NVIDIA devices may expose
// different wave sizes, while a 32-value serial reduction is negligible next
// to the memory traffic of this preprocessing path.
//
// Donor: oacpp/source/cpp/lib/oa/ml/shader/compute/quantize/computeScaleQ4.slang

import attributes;
import storage;

struct PushConstants {
	uint input_index;
	uint scale_index;
	uint count;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

static const uint BLOCK_SIZE = 32;

[kernel_name("compute_scale_q4")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID) {
	uint block_index = dispatch_thread_id.x;
	uint block_count = push.count / BLOCK_SIZE + (push.count % BLOCK_SIZE != 0 ? 1 : 0);
	if (block_index >= block_count) {
		return;
	}

	float block_max = 0.0;
	for (uint i = 0; i < BLOCK_SIZE; ++i) {
		uint element = block_index * BLOCK_SIZE + i;
		if (element < push.count) {
			float value = load_f32(storage_buffers[push.input_index], element);
			if (isfinite(value)) {
				block_max = max(block_max, abs(value));
			}
		}
	}

	float scale = block_max / 7.0;
	if (!(scale > 0.0) || !isfinite(scale)) {
		scale = 1.0;
	}
	store_f32(storage_buffers[push.scale_index], block_index, scale);
}
