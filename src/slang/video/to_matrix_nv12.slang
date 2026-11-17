// NV12 (two-plane 8-bit 4:2:0) -> normalized F32 matrix conversion.
//
// Same contract as to_matrix_i420 with an interleaved CbCr chroma plane.
// Each invocation owns one (x, y, channel) output element.

import attributes;
import storage;
import to_matrix_common;
import yuv_to_rgb;

struct PushConstants {
	uint luma_index;
	uint chroma_uv_index;
	uint output_index;
	uint out_width;
	uint out_height;
	uint output_base;
	uint channels_mode;
	uint layout_mode;
	uint interp_mode;
	uint luma_width;
	uint luma_height;
	uint chroma_width;
	uint chroma_height;
	uint color_mode;
	float src_off_x;
	float src_off_y;
	float src_scale_x;
	float src_scale_y;
	float valid_min_x;
	float valid_min_y;
	float valid_max_x;
	float valid_max_y;
	float fill_0;
	float fill_1;
	float fill_2;
	float mean_0;
	float mean_1;
	float mean_2;
	float std_0;
	float std_1;
	float std_2;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

float channel_mean(uint channel) {
	return channel == 0u ? push.mean_0 : channel == 1u ? push.mean_1 : push.mean_2;
}

float channel_std(uint channel) {
	return channel == 0u ? push.std_0 : channel == 1u ? push.std_1 : push.std_2;
}

float channel_fill(uint channel) {
	return channel == 0u ? push.fill_0 : channel == 1u ? push.fill_1 : push.fill_2;
}

[kernel_name("to_matrix_nv12")]
[domain("video")]
[variant("semiplanar_420_8bit")]
[dtype("u8")]
[status("experimental")]
[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint out_x = thread_id.x;
	uint out_y = thread_id.y;
	uint channel = thread_id.z;
	uint channel_count = push.channels_mode == CHANNELS_LUMA ? 1u : 3u;
	if (out_x >= push.out_width || out_y >= push.out_height || channel >= channel_count) return;

	float px = float(out_x) + 0.5f;
	float py = float(out_y) + 0.5f;
	bool inside = px >= push.valid_min_x && px < push.valid_max_x && py >= push.valid_min_y
		&& py < push.valid_max_y;

	float value;
	if (!inside) {
		value = channel_fill(channel);
	} else {
		float sx = px * push.src_scale_x + push.src_off_x;
		float sy = py * push.src_scale_y + push.src_off_y;
		float luma_code = to_matrix_sample_plane(
			storage_buffers[push.luma_index],
			sx,
			sy,
			push.luma_width,
			push.luma_height,
			push.interp_mode
		);
		float luma = to_matrix_scale_luma(luma_code, push.color_mode);
		if (push.channels_mode == CHANNELS_LUMA) {
			value = luma;
		} else {
			float chroma_x = (sx - 0.5f) * 0.5f;
			float chroma_y = (sy - 0.5f) * 0.5f;
			float cb_code = to_matrix_sample_nv12_chroma(
				storage_buffers[push.chroma_uv_index],
				chroma_x,
				chroma_y,
				push.chroma_width,
				push.chroma_height,
				0u,
				push.interp_mode
			);
			float cr_code = to_matrix_sample_nv12_chroma(
				storage_buffers[push.chroma_uv_index],
				chroma_x,
				chroma_y,
				push.chroma_width,
				push.chroma_height,
				1u,
				push.interp_mode
			);
			float cb = to_matrix_scale_chroma(cb_code, push.color_mode);
			float cr = to_matrix_scale_chroma(cr_code, push.color_mode);
			if (push.channels_mode == CHANNELS_YUV) {
				value = channel == 0u ? luma : channel == 1u ? cb + 0.5f : cr + 0.5f;
			} else {
				float3 rgb = yuv_to_rgb(float3(luma, cb, cr), push.color_mode >> 1);
				uint component = push.channels_mode == CHANNELS_BGR ? 2u - channel : channel;
				value = component == 0u ? rgb.x : component == 1u ? rgb.y : rgb.z;
			}
		}
	}

	float result = (value - channel_mean(channel)) / channel_std(channel);
	uint index = push.layout_mode == 0u
		? push.output_base + (channel * push.out_height + out_y) * push.out_width + out_x
		: push.output_base + (out_y * push.out_width + out_x) * channel_count + channel;
	store_f32(storage_buffers[push.output_index], index, result);
}
