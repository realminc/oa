import Reductions;
import Storage;

struct PushConstants {
	uint x_idx;
	uint weight_idx;
	uint out_idx;
	uint rows;
	uint cols;
	float eps;
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
	uint row = gid.x;
	if (row >= pc.rows) return;

	uint lane = tid.x;
	uint base = row * pc.cols;

	float inv_rms = InvRms(heap[pc.x_idx], base, pc.cols, pc.eps, lane);

	for (uint i = lane; i < pc.cols; i += 256) {
		OaStore(heap[pc.out_idx], base + i, OaLoad(heap[pc.x_idx], base + i) * inv_rms * OaLoad(heap[pc.weight_idx], i));
	}
}
