// EmpyrealmDt — fused Empyrealm per-token dt term (forward).
//
// 1:1 copy of Mamba3Dt, renamed for namespace separation. Collapses the
// preprocess chain that built DT from dd_dt + dt_bias:
//   Softplus + ClampMin + ClampMax  ==  3 elementwise dispatches  →  ONE kernel.
//
//   sp   = softplus(x) = max(x,0) + log1p(e^-|x|)    (numerically stable)
//   DT   = clamp(sp, dt_min, dt_max)
//
// x, out are [B*S, H] row-major (flat); one thread per element.
// The broadcast Add (dd_dt + dt_bias) is kept as a separate op so the
// autograd tape handles the dt_bias gradient reduction.

import Storage;
import SsmMath;

struct PushConstants {
	uint x_idx;     // bufs[0]  x     [B*S, H]  (dd_dt + dt_bias)
	uint out_idx;   // bufs[1]  DT    [B*S, H]
	uint count;     // B*S*H elements
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

	float x = OaLoad(heap[pc.x_idx], idx);
	float dt = SsmSoftplusClamp(x, pc.dt_min, pc.dt_max);

	OaStore(heap[pc.out_idx], idx, dt);
}
