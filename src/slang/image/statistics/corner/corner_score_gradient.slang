import attributes;
import storage;

struct PushConstants {
	uint input_index, products_index;
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

// Second-moment products for corner_score: per pixel writes Ix*Ix, Iy*Iy,
// Ix*Iy at consecutive f32 slots (index*3 + {0,1,2}) in the products buffer.
[kernel_name("corner_score_gradient")]
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
	uint index = (plane * push.height * push.width + y * push.width + x) * 3;
	RWByteAddressBuffer products = storage_buffers[push.products_index];
	store_f32(products, index, gx * gx);
	store_f32(products, index + 1, gy * gy);
	store_f32(products, index + 2, gx * gy);
}
