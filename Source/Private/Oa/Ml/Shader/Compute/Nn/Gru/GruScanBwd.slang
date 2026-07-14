// GruScanBwd — BPTT recurrence scan in ONE dispatch (backward of GruScan).
//
// One workgroup per batch; loops t = S-1 downto 0, maintaining dH (the gradient
// w.r.t. h^{(t+1)} contributed by later timesteps) in groupshared. Produces
// d_gates_i [B*S,3H] and d_gates_h [B*S,3H]; the weight/bias gradients are then
// one LinearWeightBiasBwd(hprev, d_gates_h) call (it sums over all B*S rows =
// over t and b), so no separate reduction kernel is needed.
//
// Forward:  h^{(t+1)} = (1-z)*n + z*h^{(t)},  n = tanh(ic + r*hn),
//           r = sigmoid(ir+hr),  z = sigmoid(iz+hz),
//           gates_h = W_hh @ h^{(t)} + b_hh.
// Given dO_t = dL/dh^{(t+1)} (= upstream output grad + recurrent dH):
//   dn = dO*(1-z)        dz = dO*(hp - n)        dhp_direct = dO*z
//   dnTanh = dn*(1-n^2)  dr = dnTanh*hn          dhn = dnTanh*r
//   dsr = dr*r*(1-r)     dsz = dz*z*(1-z)
//   d_gates_i = (dsr, dsz, dnTanh)   d_gates_h = (dsr, dsz, dhn)
//   dH_prev = dhp_direct + d_gates_h @ W_hh        // flows to timestep t-1

import Activations;
import Storage;

struct PushConstants {
	uint dout_idx;       // [B*S, H] grad w.r.t. out (h^{(t+1)})
	uint gates_i_idx;    // [B*S, 3H]
	uint hprev_idx;      // [B*S, H] (h^{(t)}, saved by the forward)
	uint w_hh_idx;       // [3H, H]
	uint b_hh_idx;       // [3H] (present iff has_bias; unused in the math, kept for parity)
	uint d_gates_i_idx;  // [B*S, 3H] out
	uint d_gates_h_idx;  // [B*S, 3H] out
	uint hidden_size;    // H
	uint seq_len;        // S
	uint batch;          // B
	uint has_bias;       // 1 if b_hh present
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

groupshared float dHShared[1024];       // recurrent grad flowing back [H]
groupshared float gatesHShared[3072];   // recomputed gates_h [3H]
groupshared float hpShared[1024];       // h^{(t)} loaded [H] (reused for dhp_direct)
groupshared float dGatesHShared[3072];  // d_gates_h [3H]

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID) {
	uint b = gid.x;
	uint lane = ltid.x;
	uint H = pc.hidden_size;
	uint S = pc.seq_len;
	uint totalGates = 3u * H;
	if (b >= pc.batch) return;

	// dH = 0 (nothing beyond the last timestep contributes)
	for (uint j0 = lane; j0 < H; j0 += 256) dHShared[j0] = 0.0f;
	GroupMemoryBarrierWithGroupSync();

	for (int t = int(S) - 1; t >= 0; --t) {
		uint tu = uint(t);

		// load h^{(t)} into shared
		uint hpBase = (b * S + tu) * H;
		for (uint j1 = lane; j1 < H; j1 += 256)
			hpShared[j1] = OaLoad(heap[pc.hprev_idx], hpBase + j1);
		GroupMemoryBarrierWithGroupSync();

		// recompute gates_h = W_hh @ h^{(t)} + b_hh (cooperative over 3H)
		for (uint idx = lane; idx < totalGates; idx += 256) {
			uint m = idx;
			uint wRow = m * H;
			float sum = (pc.has_bias != 0u) ? OaLoad(heap[pc.b_hh_idx], m) : 0.0f;
			for (uint i = 0; i < H; ++i)
				sum += hpShared[i] * OaLoad(heap[pc.w_hh_idx], wRow + i);
			gatesHShared[idx] = sum;
		}
		GroupMemoryBarrierWithGroupSync();

		// pointwise backward -> d_gates_i[t], d_gates_h[t]; stash dhp_direct in hpShared
		uint giBase = (b * S + tu) * totalGates;
		uint outBase = (b * S + tu) * H;
		for (uint j2 = lane; j2 < H; j2 += 256) {
			float ir = OaLoad(heap[pc.gates_i_idx], giBase + j2);
			float iz = OaLoad(heap[pc.gates_i_idx], giBase + H + j2);
			float ic = OaLoad(heap[pc.gates_i_idx], giBase + 2u * H + j2);
			float hr = gatesHShared[j2];
			float hz = gatesHShared[H + j2];
			float hn = gatesHShared[2u * H + j2];
			float hp = hpShared[j2];

			float r = Sigmoid(ir + hr);
			float z = Sigmoid(iz + hz);
			float n = tanh(ic + r * hn);

			float dO = OaLoad(heap[pc.dout_idx], outBase + j2) + dHShared[j2];
			float dn = dO * (1.0 - z);
			float dz = dO * (hp - n);
			float dhpDirect = dO * z;
			float dnTanh = dn * (1.0 - n * n);
			float dr = dnTanh * hn;
			float dhn = dnTanh * r;
			float dsr = dr * r * (1.0 - r);
			float dsz = dz * z * (1.0 - z);

			OaStore(heap[pc.d_gates_i_idx], giBase + j2, dsr);
			OaStore(heap[pc.d_gates_i_idx], giBase + H + j2, dsz);
			OaStore(heap[pc.d_gates_i_idx], giBase + 2u * H + j2, dnTanh);
			dGatesHShared[j2] = dsr;
			dGatesHShared[H + j2] = dsz;
			dGatesHShared[2u * H + j2] = dhn;
			hpShared[j2] = dhpDirect;  // reuse: accumulator for dH_prev
		}
		GroupMemoryBarrierWithGroupSync();

		// recurrent term: dH_prev[j] = dhp_direct[j] + sum_m d_gates_h[m]*W_hh[m*H+j]
		for (uint j3 = lane; j3 < H; j3 += 256) {
			float sum = hpShared[j3];
			for (uint m = 0; m < totalGates; ++m)
				sum += dGatesHShared[m] * OaLoad(heap[pc.w_hh_idx], m * H + j3);
			dHShared[j3] = sum;
		}
		// write d_gates_h[t] to global [B*S, 3H] for the weight-grad GEMM
		for (uint idx2 = lane; idx2 < totalGates; idx2 += 256)
			OaStore(heap[pc.d_gates_h_idx], giBase + idx2, dGatesHShared[idx2]);
		GroupMemoryBarrierWithGroupSync();  // dHShared ready for timestep t-1
	}
}
