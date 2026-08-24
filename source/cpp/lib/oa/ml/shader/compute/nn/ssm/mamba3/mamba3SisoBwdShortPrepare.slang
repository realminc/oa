// Short Mamba-3 backward, stage 1: prepare inclusive rotary Q/K.
//
// One invocation owns a (batch, head) sequence.  The short route is restricted
// to L<=16, N<=32 and A<=8, so keeping the running angle and one Q/K row private
// is cheaper than reserving a 32 KiB workgroup history.

import storage;
import activations;

static const uint THREADS = 64u;
static const uint MAXN = 32u;
static const uint MAXA = 8u;

struct PushConstants {
	uint c_idx;
	uint b_idx;
	uint dt_idx;
	uint trap_idx;
	uint adt_idx;
	uint angle_idx;
	uint cbias_idx;
	uint bbias_idx;
	uint crot_idx;
	uint brot_idx;
	uint theta_idx;
	uint qk_idx;
	uint decay_idx;
	uint gamma_idx;
	uint scale_idx;
	uint B;
	uint L;
	uint H;
	uint G;
	uint N;
	uint A;
};
[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(THREADS, 1, 1)]
void main(uint3 dispatchId : SV_DispatchThreadID) {
	uint bh = dispatchId.x;
	if (bh >= pc.B * pc.H) return;
	uint h = bh % pc.H;
	uint b = bh / pc.H;
	uint g = h / (pc.H / pc.G);

	float theta[MAXA];
	for (uint k = 0u; k < pc.A; ++k) theta[k] = 0.0f;

	for (uint t = 0u; t < pc.L; ++t) {
		uint tokBH = (b * pc.L + t) * pc.H + h;
		uint qkBase = tokBH * pc.N;
		uint inputBase = ((b * pc.L + t) * pc.G + g) * pc.N;
		uint angBase = (b * pc.L + t) * pc.A;
		uint thetaBase = (bh * pc.L + t) * pc.A;
		float dt = oaLoadF32(heap[pc.dt_idx], tokBH);
		float trap = Sigmoid(oaLoadF32(heap[pc.trap_idx], tokBH));
		float gamma = dt * trap;
		float scale = gamma;
		if (t + 1u < pc.L) {
			uint next = tokBH + pc.H;
			float dtNext = oaLoadF32(heap[pc.dt_idx], next);
			float trapNext = Sigmoid(oaLoadF32(heap[pc.trap_idx], next));
			scale += dtNext * (1.0f - trapNext);
		}
		oaStoreF32(heap[pc.decay_idx], tokBH,
			exp(oaLoadF32(heap[pc.adt_idx], tokBH)));
		oaStoreF32(heap[pc.gamma_idx], tokBH, gamma);
		oaStoreF32(heap[pc.scale_idx], tokBH, scale);

		for (uint k = 0u; k < pc.A; ++k) {
			theta[k] += oaLoadF32(heap[pc.angle_idx], angBase + k) * dt;
			oaStoreF32(heap[pc.theta_idx], thetaBase + k, theta[k]);
		}

		float qk = 0.0f;
		for (uint k = 0u; k < pc.A; ++k) {
			uint i0 = 2u * k;
			uint i1 = i0 + 1u;
			float cs = cos(theta[k]);
			float sn = sin(theta[k]);
			float c0 = oaLoadF32(heap[pc.c_idx], inputBase + i0)
				+ oaLoadF32(heap[pc.cbias_idx], h * pc.N + i0);
			float c1 = oaLoadF32(heap[pc.c_idx], inputBase + i1)
				+ oaLoadF32(heap[pc.cbias_idx], h * pc.N + i1);
			float b0 = oaLoadF32(heap[pc.b_idx], inputBase + i0)
				+ oaLoadF32(heap[pc.bbias_idx], h * pc.N + i0);
			float b1 = oaLoadF32(heap[pc.b_idx], inputBase + i1)
				+ oaLoadF32(heap[pc.bbias_idx], h * pc.N + i1);
			float cr0 = c0 * cs - c1 * sn;
			float cr1 = c0 * sn + c1 * cs;
			float br0 = b0 * cs - b1 * sn;
			float br1 = b0 * sn + b1 * cs;
			oaStoreF32(heap[pc.crot_idx], qkBase + i0, cr0);
			oaStoreF32(heap[pc.crot_idx], qkBase + i1, cr1);
			oaStoreF32(heap[pc.brot_idx], qkBase + i0, br0);
			oaStoreF32(heap[pc.brot_idx], qkBase + i1, br1);
			qk += cr0 * br0 + cr1 * br1;
		}
		for (uint n = 2u * pc.A; n < pc.N; ++n) {
			float cv = oaLoadF32(heap[pc.c_idx], inputBase + n)
				+ oaLoadF32(heap[pc.cbias_idx], h * pc.N + n);
			float bv = oaLoadF32(heap[pc.b_idx], inputBase + n)
				+ oaLoadF32(heap[pc.bbias_idx], h * pc.N + n);
			oaStoreF32(heap[pc.crot_idx], qkBase + n, cv);
			oaStoreF32(heap[pc.brot_idx], qkBase + n, bv);
			qk += cv * bv;
		}
		oaStoreF32(heap[pc.qk_idx], tokBH, qk);
	}
}
