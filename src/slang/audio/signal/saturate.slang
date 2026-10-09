import storage;
import attributes;

struct PushConstants {
	uint input_index;
	uint output_index;
	uint count;
	float drive_db;
	float mix;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("saturate")]
[domain("audio")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint index = thread_id.x;
	if (index >= push.count) return;
	float dry = load_f32(storage_buffers[push.input_index], index);
	float drive = exp2(push.drive_db / 6.020599913279624f);
	float wet = tanh(dry * drive);
	store_f32(storage_buffers[push.output_index], index, dry + (wet - dry) * push.mix);
}
