import attributes;
import storage;

struct PushConstants {
	uint input_index, output_index, batch_size, channels, height, width, bins;
	float lower, upper;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("calc_hist")]
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
	if (!(value >= push.lower && value < push.upper)) return;
	uint bucket = min(
		uint((value - push.lower) * (float(push.bins) / (push.upper - push.lower))),
		push.bins - 1u);
	uint ignored;
	storage_buffers[push.output_index].InterlockedAdd(
		(plane * push.bins + bucket) * 4u, 1u, ignored);
}
