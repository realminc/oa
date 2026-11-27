import attributes;
import storage;

struct PushConstants {
	uint input_index, accumulator_index, output_index;
	uint batch_size, channels, height, width;
	float alpha;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("accumulate_weighted")]
[domain("image")]
[variant("nchw_or_chw")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint x = thread_id.x, y = thread_id.y, plane = thread_id.z;
	if (x >= push.width || y >= push.height || plane >= push.batch_size * push.channels) return;
	uint element = (plane * push.height + y) * push.width + x;
	float blended = load_f32(storage_buffers[push.input_index], element) * push.alpha
		+ load_f32(storage_buffers[push.accumulator_index], element) * (1.0 - push.alpha);
	store_f32(storage_buffers[push.output_index], element, blended);
}
