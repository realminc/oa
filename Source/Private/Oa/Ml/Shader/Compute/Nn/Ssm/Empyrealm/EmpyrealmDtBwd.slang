// EmpyrealmDtBwd — fused Empyrealm dt backward.
//
// 1:1 copy of Mamba3DtBwd, renamed for namespace separation.
//
// Forward:  sp = softplus(x);  DT = clamp(sp, dt_min, dt_max)
// Given dOut = dL/dDT, recompute and emit the input grad.
//
//   mask  = (dt_min <= sp) and (sp <= dt_max)   (1 if clamp didn't fire, else 0)
//   dX    = dOut * mask * sigmoid(x)
//         = dOut * mask * (1 - exp(-sp))          (stable: sp >= 0)
//
// x, dOut, dX are [B*S, H] flat; one thread per element.

import Storage;

struct PushConstants {
	uint x_idx;      // bufs[0]  x     [B*S, H]
	uint dout_idx;   // bufs[1]  dL/dDT [B*S, H]
	uint dx_idx;     // bufs[2]  dL/dx  [B*S, H]  (write)
	uint count;
	float dt_min;
	float dt_max;
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint idx = tid.x;
	if (idx >= pc.count) return;

	float x  = OaLoad(heap[pc.x_idx], idx);
	float dO = OaLoad(heap[pc.dout_idx], idx);

	float sp   = max(x, 0.0f) + log(1.0f + exp(-abs(x)));
	float mask = (sp >= pc.dt_min && sp <= pc.dt_max) ? 1.0f : 0.0f;
	float sig  = 1.0f - exp(-sp);   // sigmoid(x) via softplus output

	OaStore(heap[pc.dx_idx], idx, dO * mask * sig);
}
