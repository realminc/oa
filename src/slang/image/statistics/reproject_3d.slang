import attributes;
import storage;

struct PushConstants {
	uint disparity_index, q_index, output_index;
	uint batch_size, channels, height, width;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

// reprojectImageTo3D-style per-pixel transform. For each pixel it
// evaluates (X, Y, Z, W) = Q * [x, y, disparity, 1] with the 4x4
// matrix Q stored row-major in the bound F32 buffer, then writes
// the three normalized coordinates X/W, Y/W, Z/W into the output's
// three channels. The homogeneous divide is unguarded to match the
// donor semantics; degenerate Q rows surface as inf/NaN.
[kernel_name("reproject_3d")]
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
	RWByteAddressBuffer disparity_buffer = storage_buffers[push.disparity_index];
	RWByteAddressBuffer q_buffer = storage_buffers[push.q_index];
	RWByteAddressBuffer output_buffer = storage_buffers[push.output_index];
	uint plane_size = push.height * push.width;
	uint in_index = plane * plane_size + y * push.width + x;
	float d = load_f32(disparity_buffer, in_index);
	float px = float(x), py = float(y);
	float w = load_f32(q_buffer, 12u) * px
		+ load_f32(q_buffer, 13u) * py
		+ load_f32(q_buffer, 14u) * d
		+ load_f32(q_buffer, 15u);
	float inv_w = 1.0f / w;
	uint out_base = plane * 3u * plane_size + y * push.width + x;
	for (uint channel = 0u; channel < 3u; ++channel) {
		float coordinate = load_f32(q_buffer, channel * 4u) * px
			+ load_f32(q_buffer, channel * 4u + 1u) * py
			+ load_f32(q_buffer, channel * 4u + 2u) * d
			+ load_f32(q_buffer, channel * 4u + 3u);
		store_f32(output_buffer, out_base + channel * plane_size, coordinate * inv_w);
	}
}
