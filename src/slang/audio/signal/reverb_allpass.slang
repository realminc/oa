import storage;
import attributes;

struct PushConstants {
	uint input_index;
	uint output_index;
	uint channels;
	uint samples;
	uint delay_samples;
	float gain;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("reverb_allpass")]
[domain("audio")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint sequence = thread_id.x;
	if (sequence >= push.channels * push.delay_samples) return;
	uint channel = sequence / push.delay_samples;
	uint residue = sequence - channel * push.delay_samples;
	uint base = channel * push.samples;
	float delayed_input = 0.0f, delayed_output = 0.0f;
	for (uint sample = residue; sample < push.samples; sample += push.delay_samples) {
		float input = load_f32(storage_buffers[push.input_index], base + sample);
		float output = -push.gain * input + delayed_input + push.gain * delayed_output;
		store_f32(storage_buffers[push.output_index], base + sample, output);
		delayed_input = input; delayed_output = output;
	}
}
