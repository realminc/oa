import attributes;
import storage;

struct PushConstants {
	uint input_index, output_index, batch_size, height, width, colormap;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

float3 hsv_wheel(float hue) {
	float sector = hue / 60.0;
	float x = 1.0 - abs(fmod(sector, 2.0) - 1.0);
	if (sector < 1.0) return float3(1.0, x, 0.0);
	if (sector < 2.0) return float3(x, 1.0, 0.0);
	if (sector < 3.0) return float3(0.0, 1.0, x);
	if (sector < 4.0) return float3(0.0, x, 1.0);
	if (sector < 5.0) return float3(x, 0.0, 1.0);
	return float3(1.0, 0.0, x);
}

[kernel_name("apply_color_map")]
[domain("image")]
[variant("nchw_or_chw")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint x = thread_id.x, y = thread_id.y, plane = thread_id.z;
	if (x >= push.width || y >= push.height || plane >= push.batch_size) return;
	uint pixel = plane * push.height * push.width + y * push.width + x;
	float value = clamp(
		load_f32(storage_buffers[push.input_index], pixel), 0.0, 1.0);
	float3 color;
	switch (push.colormap) {
	case 0u:
		color = float3(
			clamp(1.5 - abs(4.0 * value - 3.0), 0.0, 1.0),
			clamp(1.5 - abs(4.0 * value - 2.0), 0.0, 1.0),
			clamp(1.5 - abs(4.0 * value - 1.0), 0.0, 1.0));
		break;
	case 1u:
		color = float3(
			clamp(3.0 * value, 0.0, 1.0),
			clamp(3.0 * value - 1.0, 0.0, 1.0),
			clamp(3.0 * value - 2.0, 0.0, 1.0));
		break;
	case 2u:
		color = hsv_wheel(value * 360.0);
		break;
	case 3u:
		color = hsv_wheel((1.0 - value) * 240.0);
		break;
	default:
		color = float3(value, value, value);
		break;
	}
	uint stride = push.height * push.width;
	RWByteAddressBuffer output_buffer = storage_buffers[push.output_index];
	for (uint channel = 0u; channel < 3u; ++channel) {
		store_f32(output_buffer, pixel + channel * stride, color[channel]);
	}
}
