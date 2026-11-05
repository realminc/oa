// OA's native symmetric Q4 plane stores 32 values per block, one Float32 scale
// per block, and 16 payload bytes. One thread owns one aligned 32-bit payload
// word so no two invocations race on a packed byte.
//
// Donor: oacpp/source/cpp/lib/oa/ml/shader/compute/quantize/quantizeQ4.slang

import attributes;
import storage;

struct PushConstants {
	uint input_index;
	uint payload_index;
	uint scale_index;
	uint count;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

static const uint BLOCK_SIZE = 32;

[kernel_name("quantize_q4")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID) {
	uint word_index = dispatch_thread_id.x;
	uint block_count = push.count / BLOCK_SIZE + (push.count % BLOCK_SIZE != 0 ? 1 : 0);
	uint word_count = block_count * 4;
	if (word_index >= word_count) {
		return;
	}

	uint block_index = word_index / 4;
	uint block_lane = (word_index % 4) * 8;
	float scale = load_f32(storage_buffers[push.scale_index], block_index);
	bool scale_valid = isfinite(scale) && scale > 0.0;

	uint packed = 0;
	for (uint i = 0; i < 8; ++i) {
		uint element = block_index * BLOCK_SIZE + block_lane + i;
		float value = element < push.count
			? load_f32(storage_buffers[push.input_index], element)
			: 0.0;
		if (!isfinite(value)) {
			value = 0.0;
		}
		int quantized = scale_valid
			? clamp(int(round(value / scale)), -7, 7)
			: 0;
		packed |= uint(quantized + 7) << (i * 4);
	}

	store_u32(storage_buffers[push.payload_index], word_index, packed);
}
