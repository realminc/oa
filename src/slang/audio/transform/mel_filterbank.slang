import storage;
import attributes;

struct PushConstants {
	uint input_index;
	uint output_index;
	uint filterbank_index;
	uint channels;
	uint frames;
	uint freq_bins;
	uint num_mels;
	uint log_scale;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("mel_filterbank")]
[domain("audio")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(32, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint3 group_thread : SV_GroupThreadID) {
	uint mel = group_id.x * 32 + group_thread.x;
	uint frame = group_id.y, channel = group_id.z;
	if (mel >= push.num_mels || frame >= push.frames || channel >= push.channels) return;
	uint input_base = (channel * push.frames + frame) * push.freq_bins;
	uint filter_base = mel * push.freq_bins;
	float value = 0.0f;
	for (uint bin = 0; bin < push.freq_bins; ++bin) {
		value += load_f32(storage_buffers[push.input_index], input_base + bin)
			* load_f32(storage_buffers[push.filterbank_index], filter_base + bin);
	}
	if (push.log_scale != 0) value = log(value + 1e-9f);
	uint output = (channel * push.num_mels + mel) * push.frames + frame;
	store_f32(storage_buffers[push.output_index], output, value);
}
