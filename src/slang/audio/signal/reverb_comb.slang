import storage;
import attributes;

struct PushConstants {
	uint input_index;
	uint output_index;
	uint channels;
	uint in_samples;
	uint out_samples;
	uint delay_samples;
	float feedback;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("reverb_comb")]
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
	uint input_base = channel * push.in_samples;
	uint output_base = channel * push.out_samples;
	float previous = 0.0f;
	for (uint sample = residue; sample < push.out_samples; sample += push.delay_samples) {
		float input = 0.0f;
		if (sample >= push.delay_samples) {
			uint source = sample - push.delay_samples;
			if (source < push.in_samples) input = load_f32(storage_buffers[push.input_index], input_base + source);
		}
		float value = input + push.feedback * previous;
		store_f32(storage_buffers[push.output_index], output_base + sample, value);
		previous = value;
	}
}
