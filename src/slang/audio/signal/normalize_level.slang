import storage;
import attributes;

struct PushConstants {
	uint input_index;
	uint level_index;
	uint count;
	uint mode;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("normalize_level")]
[domain("audio")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(1, 1, 1)]
void main() {
	float level = 0.0f;
	if (push.mode == 0) {
		for (uint index = 0; index < push.count; ++index) {
			level = max(level, abs(load_f32(storage_buffers[push.input_index], index)));
		}
	} else {
		for (uint index = 0; index < push.count; ++index) {
			float value = load_f32(storage_buffers[push.input_index], index);
			level += value * value;
		}
		level = sqrt(level / float(push.count));
	}
	store_f32(storage_buffers[push.level_index], 0, max(level, 1.17549435e-38f));
}
