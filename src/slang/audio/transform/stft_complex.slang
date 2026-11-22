import storage;
import attributes;
import fft;

struct PushConstants {
	uint input_index;
	uint output_index;
	uint fft_size;
	uint hop_size;
	uint win_size;
	uint window;
	uint channels;
	uint samples;
	uint frames;
	uint freq_bins;
	uint center;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("stft_complex")]
[domain("audio")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint3 group_thread : SV_GroupThreadID) {
	uint frame = group_id.x, channel = group_id.z, thread = group_thread.x;
	if (frame >= push.frames || channel >= push.channels) return;
	int center_offset = push.center != 0 ? int(push.fft_size / 2) : 0;
	int frame_start = int(frame * push.hop_size) - center_offset;
	for (uint index = thread; index < push.fft_size; index += 256) {
		int sample = frame_start + int(index);
		float value = 0.0f;
		if (sample >= 0 && sample < int(push.samples)) {
			value = load_f32(storage_buffers[push.input_index], channel * push.samples + uint(sample));
		}
		fft_set_shared(index, value * audio_window_value(index, push.win_size, push.window), 0.0f);
	}
	GroupMemoryBarrierWithGroupSync();
	fft_in_place(thread, push.fft_size);
	uint output_base = (channel * push.frames + frame) * push.freq_bins * 2;
	for (uint bin = thread; bin < push.freq_bins; bin += 256) {
		store_f32(storage_buffers[push.output_index], output_base + bin * 2, fft_shared_re(bin));
		store_f32(storage_buffers[push.output_index], output_base + bin * 2 + 1, fft_shared_im(bin));
	}
}
