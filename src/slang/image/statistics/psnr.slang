import attributes;
import storage;

struct PushConstants {
	uint first_index, second_index, output_index;
	uint batch_size, channels, height, width;
	float peak_value;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("psnr")]
[domain("image")]
[variant("serial_plane")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint plane = thread_id.x;
	if (plane >= push.batch_size * push.channels) return;
	RWByteAddressBuffer first = storage_buffers[push.first_index];
	RWByteAddressBuffer second = storage_buffers[push.second_index];
	uint base = plane * push.height * push.width;
	uint count = push.height * push.width;
	float squared_error = 0.0;
	for (uint element = 0u; element < count; ++element) {
		float delta = load_f32(first, base + element) - load_f32(second, base + element);
		squared_error += delta * delta;
	}
	float mse = squared_error / float(count);
	float value = mse > 0.0
		? 10.0 * log(push.peak_value * push.peak_value / mse) / log(10.0)
		: 1.0 / 0.0;
	store_f32(storage_buffers[push.output_index], plane, value);
}
