// In-place element-wise reduce: acc[i] = op(acc[i], incoming[i])
// Used by OaCollective after staging copy to avoid CPU reduce bottleneck.
// Dispatch with DivCeil(count, 256) workgroups.

import Storage;

struct PushConstants {
	uint acc_idx;
	uint incoming_idx;
	uint count;
	uint op; // 0=Sum, 1=Max, 2=Min
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint idx = tid.x;
	if (idx >= pc.count) return;

	float acc = OaLoad(heap[pc.acc_idx], idx);
	float inc = OaLoad(heap[pc.incoming_idx], idx);

	float result;
	switch (pc.op) {
		case 0:  result = acc + inc; break;
		case 1:  result = max(acc, inc); break;
		case 2:  result = min(acc, inc); break;
		default: result = acc + inc; break;
	}

	OaStore(heap[pc.acc_idx], idx, result);
}
