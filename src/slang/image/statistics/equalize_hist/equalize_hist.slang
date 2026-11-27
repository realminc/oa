import attributes;
import storage;

struct PushConstants {
	uint input_index, output_index, batch_size, channels, height, width;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

uint quantize(float value) {
	if (!(value >= 0.0)) return 0u;
	if (value >= 1.0) return 255u;
	return uint(value * 256.0);
}

[kernel_name("equalize_hist")]
[domain("image")]
[variant("atomic_u32")]
[dtype("f32")]
[layout("nchw_or_chw")]
[status("experimental")]
[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint x = thread_id.x, y = thread_id.y, plane = thread_id.z;
	if (x >= push.width || y >= push.height || plane >= push.batch_size * push.channels) return;
	float value = load_f32(storage_buffers[push.input_index],
		(plane * push.height + y) * push.width + x);
	uint ignored;
	storage_buffers[push.output_index].InterlockedAdd(
		(plane * 256u + quantize(value)) * 4u, 1u, ignored);
}
