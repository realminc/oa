module fft;

// Canonical radix-two FFT schedule for OA compute kernels.
//
// Cooperative kernels share this workgroup-local transform instead of carrying
// per-operation copies. One workgroup owns each transform; barriers require
// uniform participation, so callers must guarantee every invocation in the
// workgroup reaches the calls below.
static const float FFT_PI = 3.14159265358979323846f;
static groupshared float shared_re[1024];
static groupshared float shared_im[1024];

[ForceInline]
public void fft_set_shared(uint index, float real, float imaginary) {
	shared_re[index] = real;
	shared_im[index] = imaginary;
}

[ForceInline]
public float fft_shared_re(uint index) { return shared_re[index]; }
[ForceInline]
public float fft_shared_im(uint index) { return shared_im[index]; }

public float audio_window_value(uint index, uint size, uint kind) {
	if (index >= size) return 0.0f;
	if (kind == 3) return 1.0f;
	float phase = 2.0f * FFT_PI * float(index) / float(size);
	if (kind == 1) return 0.54f - 0.46f * cos(phase);
	if (kind == 2) return 0.42f - 0.5f * cos(phase) + 0.08f * cos(2.0f * phase);
	return 0.5f * (1.0f - cos(phase));
}

void radix2_in_place(uint thread, uint size, bool inverse) {
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
	float direction = inverse ? 2.0f : -2.0f;
	for (uint length = 2; length <= size; length <<= 1) {
		uint half = length >> 1;
		for (uint pair = thread; pair < (size >> 1); pair += 256) {
			uint group = pair / half;
			uint offset = pair % half;
			uint left = group * length + offset;
			uint right = left + half;
			float angle = direction * FFT_PI * float(offset) / float(length);
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

public void fft_in_place(uint thread, uint size) {
	radix2_in_place(thread, size, false);
}

public void ifft_in_place(uint thread, uint size) {
	radix2_in_place(thread, size, true);
	float scale = 1.0f / float(size);
	for (uint index = thread; index < size; index += 256) {
		shared_re[index] *= scale;
		shared_im[index] *= scale;
	}
	GroupMemoryBarrierWithGroupSync();
}
