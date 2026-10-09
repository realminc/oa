import storage;
import attributes;

struct PushConstants {
	uint dry_index;
	uint wet_index;
	uint output_index;
	uint channels;
	uint in_samples;
	uint out_samples;
	float wet;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("reverb_mix")]
[domain("audio")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint index = thread_id.x;
	if (index >= push.channels * push.out_samples) return;
	uint channel = index / push.out_samples;
	uint sample = index - channel * push.out_samples;
	float dry = sample < push.in_samples
		? load_f32(storage_buffers[push.dry_index], channel * push.in_samples + sample)
		: 0.0f;
	float reverberated = load_f32(storage_buffers[push.wet_index], index);
	store_f32(storage_buffers[push.output_index], index, dry + (reverberated - dry) * push.wet);
}
