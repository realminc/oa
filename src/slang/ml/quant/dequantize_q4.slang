// All packed reads use aligned 32-bit addresses. A byte-address load at an
// arbitrary offset would be invalid for RWByteAddressBuffer and could read
// neighboring payload bytes differently across drivers.
//
// Donor: oacpp/source/cpp/lib/oa/ml/shader/compute/quantize/dequantizeQ4.slang

import attributes;
import storage;

struct PushConstants {
	uint payload_index;
	uint output_index;
	uint scale_index;
	uint count;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

static const uint BLOCK_SIZE = 32;

[kernel_name("dequantize_q4")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID) {
	uint element = dispatch_thread_id.x;
	if (element >= push.count) {
		return;
	}

	uint block_index = element / BLOCK_SIZE;
	uint lane = element % BLOCK_SIZE;
	float scale = load_f32(storage_buffers[push.scale_index], block_index);
	uint packed = load_u32(
		storage_buffers[push.payload_index], block_index * 4 + lane / 8);
	uint q4 = (packed >> ((lane % 8) * 4)) & 0x0F;
	float value = isfinite(scale) && scale > 0.0
		? float(int(q4) - 7) * scale
		: 0.0;
	store_f32(storage_buffers[push.output_index], element, value);
}
