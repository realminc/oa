// EmpyrealmDtAdt — fused dt + A·dt forward (Empyrealm namespace).
//
// Collapses two sequential dispatches into one:
//   DT  = clamp(softplus(dtRaw), dt_min, dt_max)
//   ADT = min(-heavy_tail(ddA), -afloor) * DT
//
// dtRaw, ddA are [B*S, H] flat; DT and ADT are written in-place to
// separate output buffers. One thread per element.

import Storage;

struct PushConstants {
	uint dtraw_idx;   // bufs[0]  dtRaw  [B*S, H]  (dd_dt + dt_bias)
	uint dda_idx;     // bufs[1]  dd_A   [B*S, H]
	uint dt_idx;      // bufs[2]  DT     [B*S, H]  (write)
	uint adt_idx;     // bufs[3]  ADT    [B*S, H]  (write)
	uint count;       // B*S*H elements
	float dt_min;
	float dt_max;
	float afloor;
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint idx = tid.x;
	if (idx >= pc.count) return;

	float x = OaLoad(heap[pc.dtraw_idx], idx);

	float sp = max(x, 0.0f) + log(1.0f + exp(-abs(x)));
	float dt = clamp(sp, pc.dt_min, pc.dt_max);
	OaStore(heap[pc.dt_idx], idx, dt);

	float a = OaLoad(heap[pc.dda_idx], idx);
	float h = (a >= 0.0f) ? (1.0f + a) : (1.0f / (1.0f - a));
	float aTok = min(-h, -pc.afloor);
	OaStore(heap[pc.adt_idx], idx, aTok * dt);
}
