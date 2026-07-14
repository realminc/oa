// GruCellPointwiseBwd — fused GRU pointwise backward.
//
// Given dOut = dL/dh_new and the forward gate projections, computes gradients
// w.r.t. gates_i, gates_h (each [B, 3H]) and the previous hidden h_prev ([B, H]).
//
// Forward:  h_new = (1-z)*n + z*h_prev,  n = tanh(in + r*hn),
//           r = sigmoid(ir+hr),  z = sigmoid(iz+hz)
// Backward (chain rule):
//   dn  = dOut*(1-z)            din = dn*(1-n^2)
//   dz  = dOut*(h_prev - n)     dr  = dn*(1-n^2)*hn     dhn = dn*(1-n^2)*r
//   dh_prev = dOut*z
//   dir = dhr = dr*r*(1-r)      diz = dhz = dz*z*(1-z)

import Activations;
import Storage;

struct PushConstants {
	uint gates_i_idx;
	uint gates_h_idx;
	uint hidden_idx;     // h_prev [B, H]
	uint dout_idx;       // dL/dh_new [B, H]
	uint d_gates_i_idx;  // [B, 3H] (or [B*T, 3H] with time_offset)
	uint d_gates_h_idx;  // [B, 3H]
	uint d_hidden_idx;   // [B, H]
	uint hidden_size;    // H
	uint count;          // B * H
	uint time_offset;    // timestep offset into gates_i / d_gates_i (in rows of 3H)
	uint batch_stride;   // row stride between batches in gates_i (T for batch-major, 1 for contiguous)
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint idx = tid.x;
	if (idx >= pc.count) return;

	uint H = pc.hidden_size;
	uint b = idx / H;
	uint j = idx - b * H;
	uint gi = (b * pc.batch_stride + pc.time_offset) * (3u * H) + j; // gates_i: batch-major [B*T, 3H]
	uint gh3 = b * (3u * H) + j;    // gates_h: [B, 3H] (no time dimension)
	uint gh = b * H + j;            // hidden / dout / d_hidden: [B, H]

	// Recompute forward intermediates.
	float ir = OaLoad(heap[pc.gates_i_idx], gi);
	float iz = OaLoad(heap[pc.gates_i_idx], gi + H);
	float ic = OaLoad(heap[pc.gates_i_idx], gi + 2u * H);
	float hr = OaLoad(heap[pc.gates_h_idx], gh3);
	float hz = OaLoad(heap[pc.gates_h_idx], gh3 + H);
	float hn = OaLoad(heap[pc.gates_h_idx], gh3 + 2u * H);
	float hp = OaLoad(heap[pc.hidden_idx], gh);

	float r = Sigmoid(ir + hr);
	float z = Sigmoid(iz + hz);
	float n = tanh(ic + r * hn);

	float dO = OaLoad(heap[pc.dout_idx], gh);

	float dn = dO * (1.0 - z);
	float dz = dO * (hp - n);
	float dhp = dO * z;

	float dnTanh = dn * (1.0 - n * n);   // da where a = in + r*hn
	float dr = dnTanh * hn;              // gradient w.r.t. r activation
	float dhn = dnTanh * r;

	float dsr = dr * r * (1.0 - r);      // r = sigmoid(ir+hr): dir = dhr = dsr
	float dsz = dz * z * (1.0 - z);      // z = sigmoid(iz+hz): diz = dhz = dsz

	// gates_i grads: reset=j, update=H+j, candidate=2H+j
	OaStore(heap[pc.d_gates_i_idx], gi, dsr);
	OaStore(heap[pc.d_gates_i_idx], gi + H, dsz);
	OaStore(heap[pc.d_gates_i_idx], gi + 2u * H, dnTanh);
	// gates_h grads (gates_h is [B, 3H], no time dimension)
	OaStore(heap[pc.d_gates_h_idx], gh3, dsr);
	OaStore(heap[pc.d_gates_h_idx], gh3 + H, dsz);
	OaStore(heap[pc.d_gates_h_idx], gh3 + 2u * H, dhn);
	// previous hidden grad (direct path through z*h_prev)
	OaStore(heap[pc.d_hidden_idx], gh, dhp);
}
