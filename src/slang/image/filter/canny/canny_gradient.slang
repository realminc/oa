import attributes;
import storage;

struct PushConstants {
	uint input_index, magnitude_index, direction_index;
	uint batch_size, channels, height, width;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

int reflect101(int coordinate, int extent) {
	if (coordinate < 0) return -coordinate;
	if (coordinate >= extent) return 2 * extent - 2 - coordinate;
	return coordinate;
}

float pixel_at(RWByteAddressBuffer buffer, uint plane, int x, int y) {
	uint sx = uint(reflect101(x, int(push.width)));
	uint sy = uint(reflect101(y, int(push.height)));
	return load_f32(buffer, plane * push.height * push.width + sy * push.width + sx);
}

[kernel_name("canny_gradient")]
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
	RWByteAddressBuffer input_buffer = storage_buffers[push.input_index];
	int ix = int(x), iy = int(y);
	float gx =
		-pixel_at(input_buffer, plane, ix - 1, iy - 1)
		- 2.0 * pixel_at(input_buffer, plane, ix - 1, iy)
		- pixel_at(input_buffer, plane, ix - 1, iy + 1)
		+ pixel_at(input_buffer, plane, ix + 1, iy - 1)
		+ 2.0 * pixel_at(input_buffer, plane, ix + 1, iy)
		+ pixel_at(input_buffer, plane, ix + 1, iy + 1);
	float gy =
		-pixel_at(input_buffer, plane, ix - 1, iy - 1)
		- 2.0 * pixel_at(input_buffer, plane, ix, iy - 1)
		- pixel_at(input_buffer, plane, ix + 1, iy - 1)
		+ pixel_at(input_buffer, plane, ix - 1, iy + 1)
		+ 2.0 * pixel_at(input_buffer, plane, ix, iy + 1)
		+ pixel_at(input_buffer, plane, ix + 1, iy + 1);
	float magnitude = abs(gx) + abs(gy);
	float ax = abs(gx), ay = abs(gy);
	uint direction;
	if (ay <= ax * 0.41421356) {
		direction = 0u;
	} else if (ax <= ay * 0.41421356) {
		direction = 2u;
	} else {
		direction = gx * gy > 0.0 ? 1u : 3u;
	}
	uint index = plane * push.height * push.width + y * push.width + x;
	store_f32(storage_buffers[push.magnitude_index], index, magnitude);
	store_u32(storage_buffers[push.direction_index], index, direction);
}
