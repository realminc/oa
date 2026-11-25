import attributes;
import storage;

struct PushConstants {
	uint input_index, histogram_index, output_index;
	uint batch_size, channels, height, width, bins;
	float lower, upper;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("calc_back_project")]
[domain("image")]
[variant("nchw_or_chw")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint x = thread_id.x, y = thread_id.y, plane = thread_id.z;
	if (x >= push.width || y >= push.height || plane >= push.batch_size * push.channels) return;
	float value = load_f32(storage_buffers[push.input_index],
		(plane * push.height + y) * push.width + x);
	float result = 0.0;
	if (value >= push.lower && value < push.upper) {
		uint bucket = min(
			uint((value - push.lower) * (float(push.bins) / (push.upper - push.lower))),
			push.bins - 1u);
		result = float(load_u32(storage_buffers[push.histogram_index],
			plane * push.bins + bucket));
	}
	store_f32(storage_buffers[push.output_index],
		(plane * push.height + y) * push.width + x, result);
}
