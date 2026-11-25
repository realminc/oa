import storage;
import attributes;

struct PushConstants {
	uint input_index;
	uint output_index;
	uint batch_size;
	uint channels;
	uint input_height;
	uint input_width;
	uint output_height;
	uint output_width;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("resize_area")]
[domain("image")]
[variant("nchw_or_chw")]
[dtype("f32")]
[layout("nchw_or_chw")]
[status("experimental")]
[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint output_x = thread_id.x;
	uint output_y = thread_id.y;
	uint batch_channel = thread_id.z;
	if (output_x >= push.output_width || output_y >= push.output_height ||
		batch_channel >= push.batch_size * push.channels) return;
	float scale_x = float(push.input_width) / float(push.output_width);
	float scale_y = float(push.input_height) / float(push.output_height);
	uint plane = batch_channel * push.input_height * push.input_width;
	uint output_index =
		(batch_channel * push.output_height + output_y) * push.output_width + output_x;
	// INTER_AREA degenerates to pixel replication when zooming in.
	float x0 = float(output_x) * scale_x;
	float y0 = float(output_y) * scale_y;
	float x1 = float(output_x + 1u) * scale_x;
	float y1 = float(output_y + 1u) * scale_y;
	if (scale_x < 1.0f || scale_y < 1.0f) {
		uint sx = min(uint(floor((x0 + x1) * 0.5f)), push.input_width - 1u);
		uint sy = min(uint(floor((y0 + y1) * 0.5f)), push.input_height - 1u);
		store_f32(
			storage_buffers[push.output_index],
			output_index,
			load_f32(storage_buffers[push.input_index], plane + sy * push.input_width + sx));
		return;
	}
	uint ix0 = uint(floor(x0));
	uint iy0 = uint(floor(y0));
	uint ix1 = uint(ceil(x1));
	uint iy1 = uint(ceil(y1));
	float sum = 0.0f;
	float weight = 0.0f;
	for (uint sy = iy0; sy < iy1; ++sy) {
		float wy = min(y1, float(sy + 1u)) - max(y0, float(sy));
		if (wy <= 0.0f || sy >= push.input_height) continue;
		for (uint sx = ix0; sx < ix1; ++sx) {
			float wx = min(x1, float(sx + 1u)) - max(x0, float(sx));
			if (wx <= 0.0f || sx >= push.input_width) continue;
			float w = wx * wy;
			sum += w * load_f32(
				storage_buffers[push.input_index],
				plane + sy * push.input_width + sx);
			weight += w;
		}
	}
	store_f32(
		storage_buffers[push.output_index],
		output_index,
		weight > 0.0f ? sum / weight : 0.0f);
}
