// VqAssign — vector-quantization nearest-code assignment (VQ-VAE codebook lookup).
//
// For each latent row z_e[n, :D], find the codebook entry e[k, :D] (k in [0,K))
// that minimizes squared L2 distance, then write:
//   idx[n]    = argmin_k ||z_e[n] - e[k]||^2   (int32)
//   zq[n, :]  = e[idx[n]]                        (the gathered winning code)
//
// One thread per row. This replaces the host-side argmin loop in the VQ tutorial
// so the whole VQ step stays on-GPU (no mid-step Execute/Sync round-trip). The
// op is non-differentiable (a hard assignment); the encoder still trains through
// the straight-through estimator (z_e + Detach(zq - z_e)) and the codebook
// through Gather(codebook, idx) — neither needs a gradient from this kernel.

import Storage;

struct PushConstants {
	uint ze_idx;        // z_e   [N, D]
	uint codebook_idx;  // e     [K, D]
	uint idx_idx;       // out idx [N]   (int32)
	uint zq_idx;        // out zq  [N, D]
	uint N;             // rows (latent vectors)
	uint D;             // latent dim
	uint K;             // codebook size
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint n = tid.x;
	if (n >= pc.N) return;

	uint zBase = n * pc.D;
	float best  = 3.402823466e+38F;  // FLT_MAX
	uint  bestK = 0u;
	for (uint k = 0u; k < pc.K; ++k) {
		uint eBase = k * pc.D;
		float dist = 0.0F;
		for (uint d = 0u; d < pc.D; ++d) {
			float diff = OaLoad(heap[pc.ze_idx], zBase + d)
			           - OaLoad(heap[pc.codebook_idx], eBase + d);
			dist += diff * diff;
		}
		if (dist < best) { best = dist; bestK = k; }
	}

	// idx is int32 — one 4-byte element per row.
	heap[pc.idx_idx].Store(n * 4u, bestK);

	// Gather the winning code into zq[n, :].
	uint winBase = bestK * pc.D;
	for (uint d = 0u; d < pc.D; ++d) {
		OaStore(heap[pc.zq_idx], zBase + d, OaLoad(heap[pc.codebook_idx], winBase + d));
	}
}
