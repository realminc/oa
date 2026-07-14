// ML-DSA-65 batch key generation — one workgroup per keypair.
//
// Generates a (public_key, secret_key) pair from random seed.
//
// KeyGen algorithm:
//   1. seed(32 bytes) -> SHAKE-256 -> (rho[32], rho'[64], K[32])
//   2. A = ExpandA(rho)              // k*l=30 SHAKE-128 instances
//   3. (s1, s2) = ExpandS(rho')     // l+k=11 SHAKE-256 instances, eta=4
//   4. s1_hat = NTT(s1)
//   5. t = INTT(A * s1_hat) + s2
//   6. (t1, t0) = Power2Round(t)
//   7. pk = pack(rho, t1)            // 1952 bytes
//   8. tr = H(pk)                    // SHAKE-256(pk) -> 64 bytes
//   9. sk = pack(rho, K, tr, s1, s2, t0)  // 4032 bytes
//
// Like sign/verify, initial implementation uses multi-dispatch from C++.
// The primitives (NTT, SHAKE, ExpandA, ExpandS, Power2Round) are dispatched
// individually. Host handles byte packing of pk/sk.

struct PushConstants {
	uint seeds_idx;
	uint pubkeys_idx;
	uint secret_keys_idx;
	uint status_idx;
	uint count;
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
	uint keygen_idx = gid.x;
	if (keygen_idx >= pc.count) return;

	uint lane = tid.x;

	// Multi-dispatch pipeline (orchestrated by C++):
	// 1. Dispatch shake256 on seeds -> (rho, rho', K)
	// 2. Dispatch shake128 * 30 instances -> A matrix (k*l polynomials)
	// 3. Dispatch poly_sample (mode=1, eta) * 11 instances -> s1, s2
	// 4. Dispatch ntt_forward on s1 (l=5 polynomials)
	// 5. Dispatch ntt_pointwise + accumulate (A[i] * s1_hat) for each row i
	// 6. Dispatch ntt_inverse on result
	// 7. Dispatch poly_arith (add s2)
	// 8. Dispatch poly_round (Power2Round) -> t1, t0
	// 9. Host packs pk = (rho || pack(t1))
	// 10. Dispatch shake256 on pk -> tr
	// 11. Host packs sk = (rho || K || tr || pack(s1) || pack(s2) || pack(t0))
	//
	// WIP / ROADMAP: none of the above is implemented. This writes a status word
	// only — it produces NO keypair. Unwired; ML-DSA keygen runs on the CPU
	// (liboqs, Sign.cpp).
	if (lane == 0) {
		heap[pc.status_idx].Store<uint>(keygen_idx * 4, 1);
	}
}
