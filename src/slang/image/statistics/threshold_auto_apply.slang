import attributes;
import storage;

struct PushConstants {
	uint input_index, thresholds_index, output_index, batch_size, channels;
	uint height, width;
	float max_value;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

uint quantize(float value) {
	return min(uint(clamp(value, 0.0, 1.0) * 256.0), 255u);
}

[kernel_name("threshold_auto_apply")]
[domain("image")]
[variant("nchw_or_chw")]
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
	float level = load_f32(storage_buffers[push.thresholds_index], plane);
	float result = quantize(value) > uint(level + 0.5) ? push.max_value : 0.0;
	store_f32(storage_buffers[push.output_index],
		(plane * push.height + y) * push.width + x, result);
}
