import attributes;
import storage;

struct PushConstants {
	uint input_index, kernel_x_index, kernel_y_index, output_index;
	uint batch_size, channels, height, width;
	uint operation, border_mode, kernel_width, kernel_height, kernel_size, dx, dy;
	float p0, p1, p2, p3, border_value;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

int border_coordinate(int coordinate, uint extent) {
	if (coordinate >= 0 && coordinate < int(extent)) return coordinate;
	if (push.border_mode == 0u) return -1;
	if (push.border_mode == 1u) return clamp(coordinate, 0, int(extent) - 1);
	if (extent == 1u) return 0;
	int period = push.border_mode == 2u ? int(2u * extent) : int(2u * (extent - 1u));
	if (push.border_mode == 4u) period = int(extent);
	int folded = coordinate % period;
	if (folded < 0) folded += period;
	if (push.border_mode == 4u || folded < int(extent)) return folded;
	return push.border_mode == 2u ? period - 1 - folded : period - folded;
}

float sample_one(uint plane, int y, int x) {
	int source_x = border_coordinate(x, push.width);
	int source_y = border_coordinate(y, push.height);
	if (source_x < 0 || source_y < 0) return push.border_value;
	uint index = (plane * push.height + uint(source_y)) * push.width + uint(source_x);
	return load_f32(storage_buffers[push.input_index], index);
}

float convolve_matrix(uint plane, int y, int x) {
	float sum = 0.0f;
	int radius_y = int(push.kernel_height / 2u), radius_x = int(push.kernel_width / 2u);
	for (uint ky = 0u; ky < push.kernel_height; ++ky)
		for (uint kx = 0u; kx < push.kernel_width; ++kx)
			sum += sample_one(plane, y + int(ky) - radius_y, x + int(kx) - radius_x) *
				load_f32(storage_buffers[push.kernel_x_index], ky * push.kernel_width + kx);
	return sum;
}

float horizontal_custom(uint plane, int y, int x) {
	float sum = 0.0f;
	int radius = int(push.kernel_width / 2u);
	for (uint k = 0u; k < push.kernel_width; ++k)
		sum += sample_one(plane, y, x + int(k) - radius) *
			load_f32(storage_buffers[push.kernel_x_index], k);
	return sum;
}

float separable_custom(uint plane, int y, int x) {
	float sum = 0.0f;
	int radius = int(push.kernel_height / 2u);
	for (uint k = 0u; k < push.kernel_height; ++k) {
		int source_y = y + int(k) - radius;
		int mapped_y = border_coordinate(source_y, push.height);
		float intermediate = mapped_y < 0 ? push.border_value : horizontal_custom(plane, mapped_y, x);
		sum += intermediate * load_f32(storage_buffers[push.kernel_y_index], k);
	}
	return sum;
}

float morphology_once(uint plane, int y, int x, bool dilate) {
	float result = dilate ? -3.402823466e+38f : 3.402823466e+38f;
	int radius_y = int(push.kernel_height / 2u), radius_x = int(push.kernel_width / 2u);
	for (uint ky = 0u; ky < push.kernel_height; ++ky)
		for (uint kx = 0u; kx < push.kernel_width; ++kx) {
			float value = sample_one(plane, y + int(ky) - radius_y, x + int(kx) - radius_x);
			result = dilate ? max(result, value) : min(result, value);
		}
	return result;
}

float morphology_twice(uint plane, int y, int x, bool first_dilate) {
	bool second_dilate = !first_dilate;
	float result = second_dilate ? -3.402823466e+38f : 3.402823466e+38f;
	int radius_y = int(push.kernel_height / 2u), radius_x = int(push.kernel_width / 2u);
	for (uint ky = 0u; ky < push.kernel_height; ++ky)
		for (uint kx = 0u; kx < push.kernel_width; ++kx) {
			int source_y = y + int(ky) - radius_y, source_x = x + int(kx) - radius_x;
			int mapped_y = border_coordinate(source_y, push.height);
			int mapped_x = border_coordinate(source_x, push.width);
			float value = mapped_y < 0 || mapped_x < 0
				? push.border_value : morphology_once(plane, mapped_y, mapped_x, first_dilate);
			result = second_dilate ? max(result, value) : min(result, value);
		}
	return result;
}

float gaussian(uint plane, int y, int x, float sigma, uint size) {
	int radius = int(size / 2u);
	float weighted = 0.0f, weight_sum = 0.0f;
	for (uint ky = 0u; ky < size; ++ky) {
		int offset_y = int(ky) - radius;
		for (uint kx = 0u; kx < size; ++kx) {
			int offset_x = int(kx) - radius;
			float distance = float(offset_x * offset_x + offset_y * offset_y);
			float weight = exp(-distance / (2.0f * sigma * sigma));
			weighted += sample_one(plane, y + offset_y, x + offset_x) * weight;
			weight_sum += weight;
		}
	}
	return weighted / max(weight_sum, 1.0e-20f);
}

float fixed_linear(uint plane, int y, int x, uint operation) {
	float sum = 0.0f;
	for (int ky = -1; ky <= 1; ++ky)
		for (int kx = -1; kx <= 1; ++kx) {
			float coefficient = 0.0f;
			if (operation == 3u) {
				if (push.dx == 1u) coefficient = float((kx == -1 ? -1 : kx == 1 ? 1 : 0) * (ky == 0 ? 2 : 1));
				else coefficient = float((ky == -1 ? -1 : ky == 1 ? 1 : 0) * (kx == 0 ? 2 : 1));
			} else if (operation == 4u) {
				if (push.dx == 1u) coefficient = float((kx == -1 ? -1 : kx == 1 ? 1 : 0) * (ky == 0 ? 10 : 3));
				else coefficient = float((ky == -1 ? -1 : ky == 1 ? 1 : 0) * (kx == 0 ? 10 : 3));
			} else if (operation == 5u) {
				coefficient = kx == 0 && ky == 0 ? -4.0f : (abs(kx) + abs(ky) == 1 ? 1.0f : 0.0f);
			} else {
				coefficient = kx == 0 && ky == 0 ? 1.0f + 4.0f * push.p0 :
					(abs(kx) + abs(ky) == 1 ? -push.p0 : 0.0f);
			}
			sum += sample_one(plane, y + ky, x + kx) * coefficient;
		}
	return sum;
}

float neighborhood(uint plane, int y, int x, uint operation) {
	int radius = int(push.kernel_size / 2u);
	float center = sample_one(plane, y, x);
	if (operation == 13u) {
		float samples[225];
		uint count = 0u;
		for (uint ky = 0u; ky < push.kernel_size; ++ky)
			for (uint kx = 0u; kx < push.kernel_size; ++kx) {
				float value = sample_one(plane, y + int(ky) - radius, x + int(kx) - radius);
				uint position = count;
				while (position > 0u && samples[position - 1u] > value) {
					samples[position] = samples[position - 1u];
					--position;
				}
				samples[position] = value;
				++count;
			}
		return samples[count / 2u];
	}
	float weighted = 0.0f, weight_sum = 0.0f;
	for (uint ky = 0u; ky < push.kernel_size; ++ky) {
		int offset_y = int(ky) - radius;
		for (uint kx = 0u; kx < push.kernel_size; ++kx) {
			int offset_x = int(kx) - radius;
			float value = sample_one(plane, y + offset_y, x + offset_x);
			float weight = 1.0f;
			if (operation == 14u) {
				float color = value - center;
				weight = exp(-float(offset_x * offset_x + offset_y * offset_y) / (2.0f * push.p1 * push.p1) -
					color * color / (2.0f * push.p0 * push.p0));
			} else if (operation == 19u) {
				weight = exp(-float(offset_x * offset_x + offset_y * offset_y) / (2.0f * push.p2 * push.p2));
			}
			weighted += value * weight;
			weight_sum += weight;
		}
	}
	float filtered = weighted / max(weight_sum, 1.0e-20f);
	if (operation == 14u) return filtered;
	return center > filtered - push.p0 ? push.p1 : 0.0f;
}

[kernel_name("filter")]
[domain("image")]
[variant("parameterized")]
[dtype("f32")]
[layout("nchw_or_chw")]
[status("experimental")]
[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint x = thread_id.x, y = thread_id.y, plane = thread_id.z;
	if (x >= push.width || y >= push.height || plane >= push.batch_size * push.channels) return;
	float center = sample_one(plane, int(y), int(x));
	float result = center;
	if (push.operation == 0u) result = convolve_matrix(plane, int(y), int(x));
	else if (push.operation == 1u) result = separable_custom(plane, int(y), int(x));
	else if (push.operation == 2u) {
		result = 0.0f;
		int ry = int(push.kernel_height / 2u), rx = int(push.kernel_width / 2u);
		for (uint ky = 0u; ky < push.kernel_height; ++ky)
			for (uint kx = 0u; kx < push.kernel_width; ++kx)
				result += sample_one(plane, int(y) + int(ky) - ry, int(x) + int(kx) - rx);
		result /= float(push.kernel_height * push.kernel_width);
	} else if (push.operation >= 3u && push.operation <= 5u || push.operation == 12u)
		result = fixed_linear(plane, int(y), int(x), push.operation);
	else if (push.operation == 6u) result = morphology_once(plane, int(y), int(x), false);
	else if (push.operation == 7u) result = morphology_once(plane, int(y), int(x), true);
	else if (push.operation == 8u) result = morphology_twice(plane, int(y), int(x), false);
	else if (push.operation == 9u) result = morphology_twice(plane, int(y), int(x), true);
	else if (push.operation == 10u) result = morphology_once(plane, int(y), int(x), true) - morphology_once(plane, int(y), int(x), false);
	else if (push.operation == 11u) result = gaussian(plane, int(y), int(x), push.p0, push.kernel_size);
	else if (push.operation == 13u || push.operation == 14u || push.operation >= 18u)
		result = neighborhood(plane, int(y), int(x), push.operation);
	else if (push.operation == 15u) {
		float blurred = gaussian(plane, int(y), int(x), push.p0, push.kernel_size);
		result = center + push.p1 * (center - blurred);
	} else if (push.operation == 16u)
		result = center - morphology_twice(plane, int(y), int(x), false);
	else if (push.operation == 17u)
		result = morphology_twice(plane, int(y), int(x), true) - center;
	uint target = (plane * push.height + y) * push.width + x;
	store_f32(storage_buffers[push.output_index], target, result);
}
