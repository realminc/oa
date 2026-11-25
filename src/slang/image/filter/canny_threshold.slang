import attributes;
import storage;

struct PushConstants {
	uint suppressed_index, mask_index;
	uint batch_size, channels, height, width;
	float low_threshold, high_threshold;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("canny_threshold")]
[domain("image")]
[variant("nchw_or_chw")]
[dtype("u32")]
[status("experimental")]
[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint x = thread_id.x;
	uint y = thread_id.y;
	uint plane = thread_id.z;
	if (x >= push.width || y >= push.height || plane >= push.batch_size * push.channels) return;
	uint index = plane * push.height * push.width + y * push.width + x;
	float magnitude = load_f32(storage_buffers[push.suppressed_index], index);
	uint mask = magnitude >= push.high_threshold
		? 2u
		: (magnitude >= push.low_threshold ? 1u : 0u);
	store_u32(storage_buffers[push.mask_index], index, mask);
}
