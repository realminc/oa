import storage;
import attributes;

struct PushConstants {
	uint input_index;
	uint output_index;
	uint channels;
	uint samples;
	uint bins;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("waveform_envelope")]
[domain("audio")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint bin = thread_id.x;
	if (bin >= push.bins) return;
	uint width = push.samples / push.bins;
	uint remainder = push.samples % push.bins;
	uint begin = bin * width + min(bin, remainder);
	uint end = begin + width + (bin < remainder ? 1 : 0);
	float seed = begin < end ? load_f32(storage_buffers[push.input_index], begin) : 0.0f;
	float minimum = seed;
	float maximum = seed;
	for (uint channel = 0; channel < push.channels; ++channel) {
		uint base = channel * push.samples;
		for (uint sample = begin; sample < end; ++sample) {
			float value = load_f32(storage_buffers[push.input_index], base + sample);
			minimum = min(minimum, value);
			maximum = max(maximum, value);
		}
	}
	store_f32(storage_buffers[push.output_index], bin * 2, minimum);
	store_f32(storage_buffers[push.output_index], bin * 2 + 1, maximum);
}
