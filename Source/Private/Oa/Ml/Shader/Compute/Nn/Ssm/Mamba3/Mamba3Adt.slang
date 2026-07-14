// Mamba3Adt — fused Mamba-3 per-token A·dt term (forward).
//
// Collapses the preprocess chain that built ADT from the projected dd_A:
//   HeavyTailActivation (6 ops: ClampMin+Neg+ClampMin+AddScalar+Reciprocal+Add)
//   + Neg + ClampMax + Mul  ==  ~9 elementwise dispatches  →  ONE kernel.
//
//   h(x)  = heavy_tail(x) = 1 + x          if x >= 0
//                         = 1 / (1 - x)    if x <  0   (positive, C1 at 0)
//   aTok  = min(-h(ddA), -AFloor)          // clamp_max(-h, -AFloor); always < 0
//   ADT   = aTok * dt
//
// ddA, dt, out are all [B*S, H] row-major (flat); one thread per element.

import Storage;
import SsmMath;

struct PushConstants {
	uint dda_idx;   // bufs[0]  dd_A   [B*S, H]
	uint dt_idx;    // bufs[1]  dt     [B*S, H] (already softplus+clamped)
	uint out_idx;   // bufs[2]  ADT    [B*S, H]
	uint count;     // B*S*H elements
	float afloor;   // A magnitude floor (ADT capped at -afloor * dt-sign via aTok)
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

	OaStore(heap[pc.out_idx], idx, SsmAdtFromDdA(x, dt, pc.afloor));
}
