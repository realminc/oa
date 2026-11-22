import storage;
import attributes;
import fft;

struct PushConstants {
	uint input_index;
	uint scratch_index;
	uint norm_index;
	uint fft_size;
	uint hop_size;
	uint win_size;
	uint window;
	uint channels;
	uint frames;
	uint freq_bins;
	uint padded_samples;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("istft_frames")]
[domain("audio")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint3 group_thread : SV_GroupThreadID) {
	uint frame = group_id.x, channel = group_id.z, thread = group_thread.x;
	if (frame >= push.frames || channel >= push.channels) return;
	uint input_base = (channel * push.frames + frame) * push.freq_bins * 2;
	for (uint bin = thread; bin < push.fft_size; bin += 256) {
		if (bin < push.freq_bins) {
			fft_set_shared(bin,
				load_f32(storage_buffers[push.input_index], input_base + bin * 2),
				load_f32(storage_buffers[push.input_index], input_base + bin * 2 + 1));
		} else {
			uint mirror = push.fft_size - bin;
			fft_set_shared(bin,
				load_f32(storage_buffers[push.input_index], input_base + mirror * 2),
				-load_f32(storage_buffers[push.input_index], input_base + mirror * 2 + 1));
		}
	}
	GroupMemoryBarrierWithGroupSync();
	ifft_in_place(thread, push.fft_size);
	uint output_base = channel * push.padded_samples + frame * push.hop_size;
	for (uint index = thread; index < push.fft_size; index += 256) {
		float window = audio_window_value(index, push.win_size, push.window);
		uint sample = output_base + index;
		atomic_add_f32(storage_buffers[push.scratch_index], sample, fft_shared_re(index) * window);
		atomic_add_f32(storage_buffers[push.norm_index], sample, window * window);
	}
}
