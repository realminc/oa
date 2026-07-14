// EmpyrealmAdtBwd — fused Empyrealm A·dt backward.
//
// 1:1 copy of Mamba3AdtBwd, renamed for namespace separation.
//
// Forward:  h = heavy_tail(ddA);  aTok = min(-h, -afloor);  ADT = aTok * dt
// Given dOut = dL/dADT, recompute the intermediates and emit the input grads.
//
//   dDt   = dOut * aTok
//   daTok = dOut * dt
//   aTok = min(-h, -afloor):  d(aTok)/d(-h) = (-h < -afloor) ? 1 : 0  ==  (h > afloor)
//   d(-h)/dx = -h'(x),  h'(x) = 1            if x >= 0
//                              = 1/(1-x)^2   if x <  0
//   dDdA  = daTok * mask * (-h'(x))
//
// ddA, dt, dOut, dDdA, dDt are all [B*S, H] flat; one thread per element.

import Storage;

struct PushConstants {
	uint dda_idx;    // bufs[0]  dd_A   [B*S, H]
	uint dt_idx;     // bufs[1]  dt     [B*S, H]
	uint dout_idx;   // bufs[2]  dL/dADT [B*S, H]
	uint ddda_idx;   // bufs[3]  dL/dddA [B*S, H]   (write)
	uint ddt_idx;    // bufs[4]  dL/ddt  [B*S, H]   (write)
	uint count;      // B*S*H elements
	float afloor;
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint idx = tid.x;
	if (idx >= pc.count) return;

	float x  = OaLoad(heap[pc.dda_idx], idx);
	float dt = OaLoad(heap[pc.dt_idx], idx);
	float dO = OaLoad(heap[pc.dout_idx], idx);

	float h     = (x >= 0.0f) ? (1.0f + x) : (1.0f / (1.0f - x));
	float aTok  = min(-h, -pc.afloor);

	// dDt = dOut * aTok
	OaStore(heap[pc.ddt_idx], idx, dO * aTok);

	// dDdA = dOut * dt * mask * (-h'(x))
	float mask  = (h > pc.afloor) ? 1.0f : 0.0f;      // clamp_max active region
	float hp    = (x >= 0.0f) ? 1.0f : (1.0f / ((1.0f - x) * (1.0f - x)));
	float dDdA  = dO * dt * mask * (-hp);
	OaStore(heap[pc.ddda_idx], idx, dDdA);
}
