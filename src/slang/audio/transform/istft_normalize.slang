import storage;
import attributes;

struct PushConstants {
	uint scratch_index;
	uint norm_index;
	uint output_index;
	uint crop_offset;
	uint out_samples;
	uint channels;
	uint padded_samples;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("istft_normalize")]
[domain("audio")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_id : SV_DispatchThreadID) {
	uint index = dispatch_id.x;
	uint total = push.out_samples * push.channels;
	if (index >= total) return;
	uint channel = index / push.out_samples;
	uint sample = index % push.out_samples;
	uint padded = channel * push.padded_samples + push.crop_offset + sample;
	float norm = load_f32(storage_buffers[push.norm_index], padded);
	float value = 0.0f;
	if (norm > 1e-12f) {
		value = load_f32(storage_buffers[push.scratch_index], padded) / norm;
	}
	store_f32(storage_buffers[push.output_index], index, value);
}
