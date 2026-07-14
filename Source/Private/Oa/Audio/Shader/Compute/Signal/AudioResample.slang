// AudioResample — windowed-sinc sample-rate conversion with anti-aliasing.
//
// Input:  [Channels, InSamples]  F32 planar
// Output: [Channels, OutSamples] F32 planar, OutSamples = InSamples·Out/In
//
// One thread per output sample (gather). Source position is computed with
// exact rational arithmetic: the wrapper gcd-reduces InRate/OutRate before
// pushing, so r·in_rate_r below stays within u32 for all real rate pairs.
// Downsampling lowpasses at the output Nyquist (kernel scaled by the ratio);
// the kernel sum is normalized so DC is preserved exactly.

import Storage;

static const float OA_PI = 3.14159265358979323846f;

struct PushConstants {
	uint a_idx;             // input  [Channels, InSamples]
	uint out_idx;           // output [Channels, OutSamples]
	uint count;             // total output elements (Channels · OutSamples)
	uint in_rate_r;         // gcd-reduced input rate
	uint out_rate_r;        // gcd-reduced output rate
	uint filter_half_width; // sinc taps per side
	uint in_samples;        // per channel
	uint out_samples;       // per channel
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

float Sinc(float x) {
	if (abs(x) < 1e-7f) return 1.0f;
	float px = OA_PI * x;
	return sin(px) / px;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint idx = tid.x;
	if (idx >= pc.count) return;

	uint c = idx / pc.out_samples;
	uint j = idx % pc.out_samples;

	// Exact rational source position: j·in/out = n0 + fracNum/out.
	uint q    = j / pc.out_rate_r;
	uint r    = j % pc.out_rate_r;
	uint num  = r * pc.in_rate_r;
	uint n0   = q * pc.in_rate_r + num / pc.out_rate_r;
	float frac = float(num % pc.out_rate_r) / float(pc.out_rate_r);

	// Anti-alias: when downsampling, scale the sinc to the output Nyquist.
	float ratio = float(pc.out_rate_r) / float(pc.in_rate_r);
	float scale = min(1.0f, ratio);
	float halfW = float(pc.filter_half_width);

	uint channelBase = c * pc.in_samples;
	float acc = 0.0f;
	float ksum = 0.0f;

	int w = int(pc.filter_half_width);
	for (int tap = -w; tap <= w; tap++) {
		float t = float(tap) - frac;
		if (abs(t) > halfW) continue;
		float window = 0.5f * (1.0f + cos(OA_PI * t / halfW));   // Hann
		float kernel = scale * Sinc(scale * t) * window;
		ksum += kernel;

		int n = int(n0) + tap;
		if (n >= 0 && n < int(pc.in_samples)) {
			acc += OaLoad(heap[pc.a_idx], channelBase + uint(n)) * kernel;
		}
	}

	float value = ksum > 1e-7f ? acc / ksum : 0.0f;
	OaStore(heap[pc.out_idx], c * pc.out_samples + j, value);
}
