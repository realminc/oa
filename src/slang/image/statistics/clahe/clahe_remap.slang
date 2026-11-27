import attributes;
import storage;

struct PushConstants {
	uint input_index, lut_index, output_index;
	uint batch_size, channels, height, width, tile_width, tile_height, tiles_x, tiles_y;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

float lut_value(RWByteAddressBuffer lut, uint plane, uint tx, uint ty, uint bucket) {
	uint tile = (plane * push.tiles_y + ty) * push.tiles_x + tx;
	return load_f32(lut, tile * 256u + bucket);
}

[kernel_name("clahe_remap")]
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
	float value = load_f32(storage_buffers[push.input_index], element);
	uint bucket = min(uint(clamp(value, 0.0, 1.0) * 256.0), 255u);
	RWByteAddressBuffer lut = storage_buffers[push.lut_index];
	// Bilinear blend between tile lookup tables at tile-center anchors.
	float xf = (float(x) + 0.5) / float(push.tile_width) - 0.5;
	float yf = (float(y) + 0.5) / float(push.tile_height) - 0.5;
	float xa = clamp(xf, 0.0, float(push.tiles_x - 1u));
	float ya = clamp(yf, 0.0, float(push.tiles_y - 1u));
	uint tx0 = uint(xa);
	uint ty0 = uint(ya);
	uint tx1 = min(tx0 + 1u, push.tiles_x - 1u);
	uint ty1 = min(ty0 + 1u, push.tiles_y - 1u);
	float fx = xa - float(tx0);
	float fy = ya - float(ty0);
	float top = lut_value(lut, plane, tx0, ty0, bucket) * (1.0 - fx)
		+ lut_value(lut, plane, tx1, ty0, bucket) * fx;
	float bottom = lut_value(lut, plane, tx0, ty1, bucket) * (1.0 - fx)
		+ lut_value(lut, plane, tx1, ty1, bucket) * fx;
	store_f32(storage_buffers[push.output_index], element,
		top * (1.0 - fy) + bottom * fy);
}
