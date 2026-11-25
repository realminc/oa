import attributes;
import storage;

struct PushConstants {
	uint mask_index, output_index;
	uint batch_size, channels, height, width;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("canny_final")]
[domain("image")]
[variant("nchw_or_chw")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint x = thread_id.x;
	uint y = thread_id.y;
	uint plane = thread_id.z;
	if (x >= push.width || y >= push.height || plane >= push.batch_size * push.channels) return;
	uint index = plane * push.height * push.width + y * push.width + x;
	uint mask = load_u32(storage_buffers[push.mask_index], index);
	store_f32(storage_buffers[push.output_index], index, mask == 2u ? 1.0 : 0.0);
}
