import Reductions;
import Storage;

// Fused residual-add + RmsNorm: out = RmsNorm(a + b, weight)
// Also writes the pre-norm sum to residual_out for use by later layers.
// Replaces separate Add + RmsNorm dispatches in the forward pass.
//
// Launch: one workgroup per row (numthreads=256).

struct PushConstants {
	uint a_idx;
	uint b_idx;
	uint weight_idx;
	uint residual_out_idx;
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

	// Pass 1: compute residual sum and write to residual_out, accumulate sum-of-squares
	float ss = 0.0;
	for (uint i = lane; i < pc.cols; i += 256) {
		float val = OaLoad(heap[pc.a_idx], base + i) + OaLoad(heap[pc.b_idx], base + i);
		OaStore(heap[pc.residual_out_idx], base + i, val);
		ss += val * val;
	}

	float total = ReduceSum256(ss, lane);
	float inv_rms = rsqrt(total / float(pc.cols) + pc.eps);

	// Pass 2: read back the stored sum and apply RmsNorm
	for (uint i = lane; i < pc.cols; i += 256) {
		float val = OaLoad(heap[pc.residual_out_idx], base + i);
		OaStore(heap[pc.out_idx], base + i, val * inv_rms * OaLoad(heap[pc.weight_idx], i));
	}
}
