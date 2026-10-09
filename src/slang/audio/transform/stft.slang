import storage;
import attributes;

static const float PI = 3.14159265358979323846f;
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
groupshared float shared_re[1024];
groupshared float shared_im[1024];

float window_value(uint index, uint size, uint kind) {
	if (index >= size) return 0.0f;
	if (kind == 3) return 1.0f;
	float phase = 2.0f * PI * float(index) / float(size);
	if (kind == 1) return 0.54f - 0.46f * cos(phase);
	if (kind == 2) return 0.42f - 0.5f * cos(phase) + 0.08f * cos(2.0f * phase);
	return 0.5f * (1.0f - cos(phase));
}

void fft_in_place(uint thread, uint size) {
	uint bits = 0;
	for (uint value = size; value > 1; value >>= 1) ++bits;
	for (uint index = thread; index < size; index += 256) {
		uint reversed = 0, value = index;
		for (uint bit = 0; bit < bits; ++bit) {
			reversed = (reversed << 1) | (value & 1);
			value >>= 1;
		}
		if (index < reversed) {
			float real = shared_re[index], imaginary = shared_im[index];
			shared_re[index] = shared_re[reversed]; shared_im[index] = shared_im[reversed];
			shared_re[reversed] = real; shared_im[reversed] = imaginary;
		}
	}
	GroupMemoryBarrierWithGroupSync();
	for (uint length = 2; length <= size; length <<= 1) {
		uint half = length >> 1;
		for (uint pair = thread; pair < (size >> 1); pair += 256) {
			uint group = pair / half;
			uint offset = pair % half;
			uint left = group * length + offset;
			uint right = left + half;
			float angle = -2.0f * PI * float(offset) / float(length);
			float wr = cos(angle), wi = sin(angle);
			float ur = shared_re[left], ui = shared_im[left];
			float vr = shared_re[right] * wr - shared_im[right] * wi;
			float vi = shared_re[right] * wi + shared_im[right] * wr;
			shared_re[left] = ur + vr; shared_im[left] = ui + vi;
			shared_re[right] = ur - vr; shared_im[right] = ui - vi;
		}
		GroupMemoryBarrierWithGroupSync();
	}
}

[kernel_name("stft")]
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
		shared_re[index] = value * window_value(index, push.win_size, push.window);
		shared_im[index] = 0.0f;
	}
	GroupMemoryBarrierWithGroupSync();
	fft_in_place(thread, push.fft_size);
	uint output_base = (channel * push.frames + frame) * push.freq_bins;
	for (uint bin = thread; bin < push.freq_bins; bin += 256) {
		float real = shared_re[bin], imaginary = shared_im[bin];
		store_f32(storage_buffers[push.output_index], output_base + bin,
			sqrt(real * real + imaginary * imaginary));
	}
}
