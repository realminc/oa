import storage;
import attributes;

struct PushConstants {
	uint input_index, output_index, batch_size, input_channels, height, width;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("grayscale")]
[domain("image")]
[variant("rec709_rgb")]
[dtype("f32")]
[layout("nchw_or_chw")]
[status("experimental")]
[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint x = thread_id.x, y = thread_id.y, batch = thread_id.z;
	if (x >= push.width || y >= push.height || batch >= push.batch_size) return;
	uint plane = push.height * push.width;
	uint base = batch * push.input_channels * plane + y * push.width + x;
	float red = load_f32(storage_buffers[push.input_index], base);
	float green = load_f32(storage_buffers[push.input_index], base + plane);
	float blue = load_f32(storage_buffers[push.input_index], base + 2u * plane);
	store_f32(storage_buffers[push.output_index], batch * plane + y * push.width + x,
		0.2126f * red + 0.7152f * green + 0.0722f * blue);
}
