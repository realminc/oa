// Short Mamba-3 backward, stage 4: angle reverse-cumsum and dt/trap chain rule.

import storage;
import activations;
import attributes;

static const uint THREADS = 64u;
static const uint MAXA = 8u;

struct PushConstants {
	uint dt_idx;
	uint trap_idx;
	uint angle_idx;
	uint dgamma_idx;
	uint dscale_idx;
	uint dtheta_idx;
	uint ddt_idx;
	uint dtrap_idx;
	uint dangle_idx;
	uint b;
	uint l;
	uint h;
	uint a;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("mamba3_siso_backward_finalize")]
[domain("ml")]
[variant("short_finalize")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(THREADS, 1, 1)]
void main(uint3 dispatchId : SV_DispatchThreadID) {
	uint bh = dispatchId.x;
	if (bh >= push.b * push.h) return;
	uint h = bh % push.h;
	uint b = bh / push.h;
	float suffix[MAXA];
	for (uint k = 0u; k < push.a; ++k) suffix[k] = 0.0f;

	for (uint rr = 0u; rr < push.l; ++rr) {
		uint t = push.l - 1u - rr;
		uint tokBH = (b * push.l + t) * push.h + h;
		uint thetaBase = (bh * push.l + t) * push.a;
		uint angleBase = (b * push.l + t) * push.a;
		float dtTheta = 0.0f;
		float dt = load_f32(storage_buffers[push.dt_idx], tokBH);
		for (uint k = 0u; k < push.a; ++k) {
			suffix[k] += load_f32(storage_buffers[push.dtheta_idx], thetaBase + k);
			store_f32(storage_buffers[push.dangle_idx], (bh * push.l + t) * push.a + k,
				suffix[k] * dt);
			dtTheta += suffix[k] * load_f32(storage_buffers[push.angle_idx], angleBase + k);
		}
		float trap = sigmoid(load_f32(storage_buffers[push.trap_idx], tokBH));
		float dGamma = load_f32(storage_buffers[push.dgamma_idx], tokBH);
		float dDt = dGamma * trap + dtTheta;
		float dTrapRaw = dGamma * dt;
		if (t > 0u) {
			uint previous = tokBH - push.h;
			float incoming = load_f32(storage_buffers[push.dscale_idx], previous);
			dDt += incoming * (1.0f - trap);
			dTrapRaw -= incoming * dt;
		}
		store_f32(storage_buffers[push.ddt_idx], tokBH, dDt);
		store_f32(storage_buffers[push.dtrap_idx], tokBH, dTrapRaw * trap * (1.0f - trap));
	}
}





