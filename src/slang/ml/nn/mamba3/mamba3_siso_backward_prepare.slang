// Short Mamba-3 backward, stage 1: prepare inclusive rotary Q/K.
//
// One invocation owns a (batch, head) sequence.  The short route is restricted
// to L<=16, N<=32 and A<=8, so keeping the running angle and one Q/K row private
// is cheaper than reserving a 32 KiB workgroup history.

import storage;
import activations;
import attributes;

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
	uint b;
	uint l;
	uint h;
	uint g;
	uint n;
	uint a;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("mamba3_siso_backward_prepare")]
[domain("ml")]
[variant("short_prepare")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(THREADS, 1, 1)]
void main(uint3 dispatchId : SV_DispatchThreadID) {
	uint bh = dispatchId.x;
	if (bh >= push.b * push.h) return;
	uint h = bh % push.h;
	uint b = bh / push.h;
	uint g = h / (push.h / push.g);

	float theta[MAXA];
	for (uint k = 0u; k < push.a; ++k) theta[k] = 0.0f;

	for (uint t = 0u; t < push.l; ++t) {
		uint tokBH = (b * push.l + t) * push.h + h;
		uint qkBase = tokBH * push.n;
		uint inputBase = ((b * push.l + t) * push.g + g) * push.n;
		uint angBase = (b * push.l + t) * push.a;
		uint thetaBase = (bh * push.l + t) * push.a;
		float dt = load_f32(storage_buffers[push.dt_idx], tokBH);
		float trap = sigmoid(load_f32(storage_buffers[push.trap_idx], tokBH));
		float gamma = dt * trap;
		float scale = gamma;
		if (t + 1u < push.l) {
			uint next = tokBH + push.h;
			float dtNext = load_f32(storage_buffers[push.dt_idx], next);
			float trapNext = sigmoid(load_f32(storage_buffers[push.trap_idx], next));
			scale += dtNext * (1.0f - trapNext);
		}
		store_f32(storage_buffers[push.decay_idx], tokBH,
			exp(load_f32(storage_buffers[push.adt_idx], tokBH)));
		store_f32(storage_buffers[push.gamma_idx], tokBH, gamma);
		store_f32(storage_buffers[push.scale_idx], tokBH, scale);

		for (uint k = 0u; k < push.a; ++k) {
			theta[k] += load_f32(storage_buffers[push.angle_idx], angBase + k) * dt;
			store_f32(storage_buffers[push.theta_idx], thetaBase + k, theta[k]);
		}

		float qk = 0.0f;
		for (uint k = 0u; k < push.a; ++k) {
			uint i0 = 2u * k;
			uint i1 = i0 + 1u;
			float cs = cos(theta[k]);
			float sn = sin(theta[k]);
			float c0 = load_f32(storage_buffers[push.c_idx], inputBase + i0)
				+ load_f32(storage_buffers[push.cbias_idx], h * push.n + i0);
			float c1 = load_f32(storage_buffers[push.c_idx], inputBase + i1)
				+ load_f32(storage_buffers[push.cbias_idx], h * push.n + i1);
			float b0 = load_f32(storage_buffers[push.b_idx], inputBase + i0)
				+ load_f32(storage_buffers[push.bbias_idx], h * push.n + i0);
			float b1 = load_f32(storage_buffers[push.b_idx], inputBase + i1)
				+ load_f32(storage_buffers[push.bbias_idx], h * push.n + i1);
			float cr0 = c0 * cs - c1 * sn;
			float cr1 = c0 * sn + c1 * cs;
			float br0 = b0 * cs - b1 * sn;
			float br1 = b0 * sn + b1 * cs;
			store_f32(storage_buffers[push.crot_idx], qkBase + i0, cr0);
			store_f32(storage_buffers[push.crot_idx], qkBase + i1, cr1);
			store_f32(storage_buffers[push.brot_idx], qkBase + i0, br0);
			store_f32(storage_buffers[push.brot_idx], qkBase + i1, br1);
			qk += cr0 * br0 + cr1 * br1;
		}
		for (uint n = 2u * push.a; n < push.n; ++n) {
			float cv = load_f32(storage_buffers[push.c_idx], inputBase + n)
				+ load_f32(storage_buffers[push.cbias_idx], h * push.n + n);
			float bv = load_f32(storage_buffers[push.b_idx], inputBase + n)
				+ load_f32(storage_buffers[push.bbias_idx], h * push.n + n);
			store_f32(storage_buffers[push.crot_idx], qkBase + n, cv);
			store_f32(storage_buffers[push.brot_idx], qkBase + n, bv);
			qk += cv * bv;
		}
		store_f32(storage_buffers[push.qk_idx], tokBH, qk);
	}
}





