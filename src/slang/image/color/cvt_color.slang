import attributes;
import storage;

struct PushConstants {
	uint input_index, output_index, batch_size, height, width, conversion;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

float3 rgb_to_hsv(float3 color) {
	float cmax = max(color.r, max(color.g, color.b));
	float cmin = min(color.r, min(color.g, color.b));
	float delta = cmax - cmin;
	float hue = 0.0;
	if (delta > 0.0) {
		if (cmax == color.r) {
			hue = 60.0 * fmod((color.g - color.b) / delta + 6.0, 6.0);
		} else if (cmax == color.g) {
			hue = 60.0 * ((color.b - color.r) / delta + 2.0);
		} else {
			hue = 60.0 * ((color.r - color.g) / delta + 4.0);
		}
	}
	float saturation = cmax > 0.0 ? delta / cmax : 0.0;
	return float3(hue, saturation, cmax);
}

float3 hsv_to_rgb(float3 hsv) {
	float hue = fmod(hsv.x + 360.0, 360.0);
	float chroma = hsv.z * hsv.y;
	float sector = hue / 60.0;
	float x = chroma * (1.0 - abs(fmod(sector, 2.0) - 1.0));
	float3 rgb = float3(0.0, 0.0, 0.0);
	if (sector < 1.0) rgb = float3(chroma, x, 0.0);
	else if (sector < 2.0) rgb = float3(x, chroma, 0.0);
	else if (sector < 3.0) rgb = float3(0.0, chroma, x);
	else if (sector < 4.0) rgb = float3(0.0, x, chroma);
	else if (sector < 5.0) rgb = float3(x, 0.0, chroma);
	else rgb = float3(chroma, 0.0, x);
	float lift = hsv.z - chroma;
	return rgb + lift;
}

float srgb_to_linear(float value) {
	return value <= 0.04045 ? value / 12.92 : pow((value + 0.055) / 1.055, 2.4);
}

float linear_to_srgb(float value) {
	return value <= 0.0031308 ? value * 12.92 : 1.055 * pow(value, 1.0 / 2.4) - 0.055;
}

float lab_f(float value) {
	float delta = 6.0 / 29.0;
	return value > delta * delta * delta
		? pow(value, 1.0 / 3.0)
		: value / (3.0 * delta * delta) + 4.0 / 29.0;
}

float lab_f_inv(float value) {
	float delta = 6.0 / 29.0;
	return value > delta
		? value * value * value
		: 3.0 * delta * delta * (value - 4.0 / 29.0);
}

// D65/2-degree primaries; white reference Xn=0.95047, Yn=1.0, Zn=1.08883.
float3 rgb_to_lab(float3 color) {
	float r = srgb_to_linear(color.r);
	float g = srgb_to_linear(color.g);
	float b = srgb_to_linear(color.b);
	float x = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047;
	float y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b;
	float z = (0.0193339 * r + 0.1191920 * g + 0.9503041 * b) / 1.08883;
	float fx = lab_f(x);
	float fy = lab_f(y);
	float fz = lab_f(z);
	return float3(116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz));
}

float3 lab_to_rgb(float3 lab) {
	float fy = (lab.x + 16.0) / 116.0;
	float fx = fy + lab.y / 500.0;
	float fz = fy - lab.z / 200.0;
	float x = lab_f_inv(fx) * 0.95047;
	float y = lab_f_inv(fy);
	float z = lab_f_inv(fz) * 1.08883;
	float r = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z;
	float g = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z;
	float b = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z;
	return float3(linear_to_srgb(r), linear_to_srgb(g), linear_to_srgb(b));
}

[kernel_name("cvt_color")]
[domain("image")]
[variant("nchw_or_chw")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint x = thread_id.x, y = thread_id.y, plane = thread_id.z;
	if (x >= push.width || y >= push.height || plane >= push.batch_size) return;
	uint code = push.conversion;
	uint in_channels = code == 2u ? 1u : 3u;
	uint out_channels = code <= 1u ? 1u : 3u;
	uint stride = push.height * push.width;
	uint pixel = plane * stride + y * push.width + x;
	RWByteAddressBuffer input_buffer = storage_buffers[push.input_index];
	RWByteAddressBuffer output_buffer = storage_buffers[push.output_index];
	float3 source = float3(0.0, 0.0, 0.0);
	for (uint channel = 0u; channel < in_channels; ++channel) {
		source[channel] = load_f32(input_buffer, pixel + channel * stride);
	}
	// BGR-ordered codes read and write channel-swapped triplets.
	bool source_is_bgr = code == 1u || code == 4u || code == 8u || code == 12u;
	bool result_is_bgr = code == 6u || code == 10u || code == 14u;
	float3 rgb = source_is_bgr ? source.bgr : source;
	float3 result = rgb;
	switch (code) {
	case 0u:
	case 1u:
		result = float3(dot(rgb, float3(0.299, 0.587, 0.114)), 0.0, 0.0);
		break;
	case 2u:
		result = float3(source.x, source.x, source.x);
		break;
	case 3u:
	case 4u:
		result = rgb_to_hsv(rgb);
		break;
	case 5u:
	case 6u:
		result = hsv_to_rgb(rgb);
		break;
	case 7u:
	case 8u:
		result = rgb_to_lab(rgb);
		break;
	case 9u:
	case 10u:
		result = lab_to_rgb(rgb);
		break;
	case 11u:
	case 12u:
		{
			float luma = dot(rgb, float3(0.299, 0.587, 0.114));
			result = float3(
				luma,
				0.5 + (rgb.x - luma) * 0.713,
				0.5 + (rgb.z - luma) * 0.564);
		}
		break;
	case 13u:
	case 14u:
		{
			float cr = rgb.y - 0.5;
			float cb = rgb.z - 0.5;
			result = float3(
				rgb.x + 1.403 * cr,
				rgb.x - 0.714 * cr - 0.344 * cb,
				rgb.x + 1.773 * cb);
		}
		break;
	default:
		break;
	}
	float3 stored = result_is_bgr ? result.bgr : result;
	for (uint channel = 0u; channel < out_channels; ++channel) {
		store_f32(output_buffer, pixel + channel * stride, stored[channel]);
	}
}
