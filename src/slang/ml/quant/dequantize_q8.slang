// Signed bytes are extracted from aligned 32-bit payload loads. Invalid scale
// blocks produce deterministic zeros and never poison downstream Float32 work.
//
// Donor: oacpp/source/cpp/lib/oa/ml/shader/compute/quantize/dequantizeQ8.slang

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

[kernel_name("dequantize_q8")]
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
	uint packed = load_u32(
		storage_buffers[push.payload_index], block_index * 8 + lane / 4);
	uint raw = (packed >> ((lane % 4) * 8)) & 0xFF;
	int quantized = raw >= 128 ? int(raw) - 256 : int(raw);
	float scale = load_f32(storage_buffers[push.scale_index], block_index);
	float value = isfinite(scale) && scale > 0.0
		? float(quantized) * scale
		: 0.0;
	store_f32(storage_buffers[push.output_index], element, value);
}
