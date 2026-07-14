// RnnCellPointwise — fused vanilla-RNN pointwise combine (forward).
//
// Replaces the Add + Tanh dispatch pair with a single per-element kernel over
// [B, H]:
//
//   h_new = tanh(gates_i + gates_h)
//
// where gates_i = W_ih x + b_ih and gates_h = W_hh h_prev + b_hh are both
// [B, H] (the Linear projections are recorded separately; this kernel is pure
// pointwise). Same flat layout for both inputs and the output, so threads map
// 1:1 to elements with no per-row index math.

import Activations;
import Storage;

struct PushConstants {
	uint gates_i_idx;
	uint gates_h_idx;
	uint out_idx;     // h_new [B, H]
	uint count;       // B * H threads
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint idx = tid.x;
	if (idx >= pc.count) return;

	float gi = OaLoad(heap[pc.gates_i_idx], idx);
	float gh = OaLoad(heap[pc.gates_h_idx], idx);
	OaStore(heap[pc.out_idx], idx, tanh(gi + gh));
}
