import storage;
import attributes;

struct PushConstants {
	uint input_index;
	uint coefficients_index;
	uint output_index;
	uint channels;
	uint samples;
	uint section_count;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("sos_filter")]
[domain("audio")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint channel = thread_id.x;
	if (channel >= push.channels) return;
	uint base = channel * push.samples;
	for (uint sample = 0; sample < push.samples; ++sample) {
		float value = load_f32(storage_buffers[push.input_index], base + sample);
		store_f32(storage_buffers[push.output_index], base + sample, value);
	}
	for (uint section = 0; section < push.section_count; ++section) {
		uint coefficient = section * 5;
		float b0 = load_f32(storage_buffers[push.coefficients_index], coefficient + 0);
		float b1 = load_f32(storage_buffers[push.coefficients_index], coefficient + 1);
		float b2 = load_f32(storage_buffers[push.coefficients_index], coefficient + 2);
		float a1 = load_f32(storage_buffers[push.coefficients_index], coefficient + 3);
		float a2 = load_f32(storage_buffers[push.coefficients_index], coefficient + 4);
		float state0 = 0.0f, state1 = 0.0f;
		for (uint sample = 0; sample < push.samples; ++sample) {
			uint index = base + sample;
			float x = load_f32(storage_buffers[push.output_index], index);
			float y = b0 * x + state0;
			float next0 = b1 * x - a1 * y + state1;
			float next1 = b2 * x - a2 * y;
			store_f32(storage_buffers[push.output_index], index, y);
			state0 = next0; state1 = next1;
		}
	}
}
