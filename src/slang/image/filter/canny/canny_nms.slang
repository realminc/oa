import attributes;
import storage;

struct PushConstants {
	uint magnitude_index, direction_index, suppressed_index;
	uint batch_size, channels, height, width;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("canny_nms")]
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
	uint plane_size = push.height * push.width;
	uint index = plane * plane_size + y * push.width + x;
	RWByteAddressBuffer magnitude_buffer = storage_buffers[push.magnitude_index];
	RWByteAddressBuffer direction_buffer = storage_buffers[push.direction_index];
	float magnitude = load_f32(magnitude_buffer, index);
	uint direction = load_u32(direction_buffer, index);
	int dx = direction == 2u ? 0 : 1;
	int dy = direction == 0u ? 0 : 1;
	if (direction == 3u) dy = -1;
	int px = min(int(x) + dx, int(push.width) - 1);
	int py = min(int(y) + dy, int(push.height) - 1);
	int nx = max(int(x) - dx, 0);
	int ny = max(int(y) - dy, 0);
	float positive = load_f32(
		magnitude_buffer,
		plane * plane_size + uint(py) * push.width + uint(px));
	float negative = load_f32(
		magnitude_buffer,
		plane * plane_size + uint(ny) * push.width + uint(nx));
	float kept = magnitude >= positive && magnitude >= negative ? magnitude : 0.0;
	store_f32(storage_buffers[push.suppressed_index], index, kept);
}
