// ML-DSA-65 batch signing — one workgroup per signature.
//
// Signs a message using a secret key. Includes the rejection loop
// (Fiat-Shamir with Aborts — expected ~4.25 attempts for ML-DSA-65).
//
// Input: messages, secret keys, entropy (for randomized signing)
// Output: signatures, status (1 = success, 0 = exceeded max attempts)
//
// Like mldsa_verify, the initial implementation uses multi-dispatch:
//   Host orchestrates the pipeline, each dispatch handles one primitive.
//   Monolithic single-dispatch version comes after validation.
//
// Sign algorithm:
//   1. Decode sk -> (rho, K, tr, s1[l], s2[k], t0[k])
//   2. mu = H(tr || msg)                          // SHAKE-256
//   3. A = ExpandA(rho)                           // k*l SHAKE-128
//   4. s1_hat = NTT(s1), s2_hat = NTT(s2), t0_hat = NTT(t0)
//   5. loop (max MAX_ATTEMPTS):
//      a. rho' = H(K || rnd || mu)                // SHAKE-256, rnd = entropy
//      b. y = ExpandMask(rho', kappa)             // l polynomials [-gamma1,gamma1)
//      c. w = A * NTT(y)
//      d. w = INTT(w)
//      e. w1 = HighBits(w)
//      f. c_tilde = H(mu || w1)                   // SHAKE-256
//      g. c = SampleInBall(c_tilde)
//      h. c_hat = NTT(c)
//      i. z = y + INTT(c_hat * s1_hat)
//      j. if ||z||_inf >= gamma1 - beta: continue  // REJECT
//      k. r0 = LowBits(w - INTT(c_hat * s2_hat))
//      l. if ||r0||_inf >= gamma2 - beta: continue  // REJECT
//      m. ct0 = INTT(c_hat * t0_hat)
//      n. h = MakeHint(-ct0, w - c*s2 + ct0)
//      o. if sum(h) > omega(=80): continue          // REJECT
//      p. sig = encode(c_tilde, z, h)              // SUCCESS
//   6. return sig

struct PushConstants {
	uint messages_idx;
	uint secret_keys_idx;
	uint entropy_idx;
	uint sig_output_idx;
	uint status_idx;
	uint count;
	uint msg_len;
	uint max_attempts;
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

static const int Q = 8380417;
static const int QINV = 58728449;
static const uint K = 6;
static const uint L = 5;
static const int GAMMA1 = 524288;
static const int GAMMA2 = 261888;
static const int BETA = 196;
static const uint OMEGA = 80;

int montgomery_reduce(int64_t InA) {
	int t = int(InA) * QINV;
	return int((InA - int64_t(t) * int64_t(Q)) >> 32);
}

groupshared int s_work[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
	uint sign_idx = gid.x;
	if (sign_idx >= pc.count) return;

	uint lane = tid.x;

	// Initial implementation: placeholder.
	// The actual signing pipeline is orchestrated as multi-dispatch from C++:
	//
	// 1. Host decodes sk, uploads rho, K, tr, s1[l*256], s2[k*256], t0[k*256]
	// 2. Dispatch: NTT(s1), NTT(s2), NTT(t0) — reuse ntt_forward.slang
	// 3. Dispatch: ExpandA — reuse shake128.slang for k*l instances
	// 4. For each attempt (CPU loop):
	//    a. Host generates entropy, uploads
	//    b. Dispatch: ExpandMask -> y
	//    c. Dispatch: NTT(y)
	//    d. Dispatch: ntt_pointwise (A * NTT(y)) + ntt_inverse -> w
	//    e. Dispatch: poly_round (HighBits)
	//    f. Host: SHAKE hash for c_tilde (small, CPU is fine)
	//    g. Dispatch: SampleInBall -> c, NTT(c)
	//    h. Dispatch: ntt_pointwise(c_hat, s1_hat) + ntt_inverse -> cs1
	//    i. Dispatch: poly_arith (z = y + cs1)
	//    j. Host: check ||z||_inf (read back, compare)
	//    k. If reject: next attempt
	//    l. Continue with r0, hint checks...
	//    m. If all pass: encode sig
	//
	// The rejection loop is CPU-driven because the abort probability is ~76%
	// per attempt — launching a full GPU pipeline for each attempt is wasteful.
	// The GPU does the heavy math (NTT, poly ops), CPU does the control flow.
	//
	// An advanced version can run the full loop in-shader with a flag buffer.
	//
	// WIP / ROADMAP: none of the above is implemented. This writes a status word
	// only — it produces NO signature. Unwired; ML-DSA signing runs on the CPU
	// (liboqs, Sign.cpp).
	if (lane == 0) {
		heap[pc.status_idx].Store<uint>(sign_idx * 4, 1);
	}
}
