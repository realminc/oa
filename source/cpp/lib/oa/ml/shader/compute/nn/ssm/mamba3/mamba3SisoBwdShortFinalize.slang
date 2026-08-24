// Short Mamba-3 backward, stage 4: angle reverse-cumsum and dt/trap chain rule.

import storage;
import activations;

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
	uint B;
	uint L;
	uint H;
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
	float suffix[MAXA];
	for (uint k = 0u; k < pc.A; ++k) suffix[k] = 0.0f;

	for (uint rr = 0u; rr < pc.L; ++rr) {
		uint t = pc.L - 1u - rr;
		uint tokBH = (b * pc.L + t) * pc.H + h;
		uint thetaBase = (bh * pc.L + t) * pc.A;
		uint angleBase = (b * pc.L + t) * pc.A;
		float dtTheta = 0.0f;
		float dt = oaLoadF32(heap[pc.dt_idx], tokBH);
		for (uint k = 0u; k < pc.A; ++k) {
			suffix[k] += oaLoadF32(heap[pc.dtheta_idx], thetaBase + k);
			oaStoreF32(heap[pc.dangle_idx], (bh * pc.L + t) * pc.A + k,
				suffix[k] * dt);
			dtTheta += suffix[k] * oaLoadF32(heap[pc.angle_idx], angleBase + k);
		}
		float trap = Sigmoid(oaLoadF32(heap[pc.trap_idx], tokBH));
		float dGamma = oaLoadF32(heap[pc.dgamma_idx], tokBH);
		float dDt = dGamma * trap + dtTheta;
		float dTrapRaw = dGamma * dt;
		if (t > 0u) {
			uint previous = tokBH - pc.H;
			float incoming = oaLoadF32(heap[pc.dscale_idx], previous);
			dDt += incoming * (1.0f - trap);
			dTrapRaw -= incoming * dt;
		}
		oaStoreF32(heap[pc.ddt_idx], tokBH, dDt);
		oaStoreF32(heap[pc.dtrap_idx], tokBH, dTrapRaw * trap * (1.0f - trap));
	}
}
