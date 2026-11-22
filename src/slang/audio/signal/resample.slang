import storage;
import attributes;

static const float PI = 3.14159265358979323846f;
struct PushConstants {
	uint input_index;
	uint output_index;
	uint count;
	uint in_rate_r;
	uint out_rate_r;
	uint filter_half_width;
	uint in_samples;
	uint out_samples;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

float sinc(float value) {
	if (abs(value) < 1e-7f) return 1.0f;
	float phase = PI * value;
	return sin(phase) / phase;
}

[kernel_name("resample")]
[domain("audio")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint index = thread_id.x;
	if (index >= push.count) return;
	uint channel = index / push.out_samples;
	uint output_sample = index % push.out_samples;
	uint quotient = output_sample / push.out_rate_r;
	uint remainder = output_sample % push.out_rate_r;
	uint numerator = remainder * push.in_rate_r;
	uint source = quotient * push.in_rate_r + numerator / push.out_rate_r;
	float fraction = float(numerator % push.out_rate_r) / float(push.out_rate_r);
	float ratio = float(push.out_rate_r) / float(push.in_rate_r);
	float scale = min(1.0f, ratio);
	float half_width = float(push.filter_half_width);
	float sum = 0.0f;
	float kernel_sum = 0.0f;
	int width = int(push.filter_half_width);
	for (int tap = -width; tap <= width; ++tap) {
		float distance = float(tap) - fraction;
		if (abs(distance) > half_width) continue;
		float window = 0.5f * (1.0f + cos(PI * distance / half_width));
		float kernel = scale * sinc(scale * distance) * window;
		kernel_sum += kernel;
		int sample = int(source) + tap;
		if (sample >= 0 && sample < int(push.in_samples)) {
			sum += load_f32(storage_buffers[push.input_index],
				channel * push.in_samples + uint(sample)) * kernel;
		}
	}
	store_f32(storage_buffers[push.output_index], index,
		kernel_sum > 1e-7f ? sum / kernel_sum : 0.0f);
}
