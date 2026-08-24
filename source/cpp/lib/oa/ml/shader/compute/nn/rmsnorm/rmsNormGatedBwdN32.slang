// Narrow-row gated RMSNorm adjoint for at most 32 columns.
// One lane owns x/z/dy and emits the row contribution used by the existing
// deterministic parameter-gradient reductions.

import storage;
import activations;

struct PushConstants {
	uint x_idx;
	uint w_idx;
	uint bias_idx;
	uint z_idx;
	uint dy_idx;
	uint dx_idx;
	uint dz_idx;
	uint dw_idx;
	uint dbias_idx;
	uint rows;
	uint cols;
	uint groups;
	float eps;
	uint has_bias;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

groupshared float g_sum[32];

float ReduceSum32(float value, uint lane) {
	g_sum[lane] = value;
	GroupMemoryBarrierWithGroupSync();
	for (uint stride = 16; stride > 0; stride >>= 1) {
		if (lane < stride) g_sum[lane] += g_sum[lane + stride];
		GroupMemoryBarrierWithGroupSync();
	}
	return g_sum[0];
}

[shader("compute")]
[numthreads(32, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
	uint row = gid.x;
	if (row >= pc.rows) return;

	uint lane = tid.x;
	uint base = row * pc.cols;
	uint affineBase = (row % pc.groups) * pc.cols;
	bool active = lane < pc.cols;
	float x = active ? oaLoad(heap[pc.x_idx], base + lane) : 0.0;
	float invRms = rsqrt(ReduceSum32(x * x, lane) / float(pc.cols) + pc.eps);
	// Every lane must consume the first reduction before ReduceSum32 reuses the
	// same workgroup array below. Without this lifetime barrier GPU-assisted
	// validation correctly observes the next call's stores racing those reads.
	GroupMemoryBarrierWithGroupSync();
	float w = active ? oaLoad(heap[pc.w_idx], affineBase + lane) : 0.0;
	float dy = active ? oaLoad(heap[pc.dy_idx], base + lane) : 0.0;
	float z = active ? oaLoad(heap[pc.z_idx], base + lane) : 0.0;
	float sigmoidZ = Sigmoid(z);
	float gate = z * sigmoidZ;
	float dm = dy * gate;
	float c = ReduceSum32(dm * w * x, lane);

	if (active) {
		float normalized = x * invRms * w;
		if (pc.has_bias != 0) normalized += oaLoad(heap[pc.bias_idx], affineBase + lane);
		float dGate = sigmoidZ + z * sigmoidZ * (1.0 - sigmoidZ);
		oaStore(heap[pc.dx_idx], base + lane,
			invRms * (dm * w - x * c * invRms * invRms / float(pc.cols)));
		oaStore(heap[pc.dz_idx], base + lane, dy * normalized * dGate);
		oaStore(heap[pc.dw_idx], base + lane, dm * x * invRms);
		if (pc.has_bias != 0) oaStore(heap[pc.dbias_idx], base + lane, dm);
	}
}
