// Max reduction backward pass (full reduction → scalar).
//
// Forward `Max` reduces the whole tensor to a single scalar m = max(X). The
// gradient of a max w.r.t. its inputs routes the upstream scalar gradient to the
// element(s) that achieved the maximum:
//     grad_in[i] = (X[i] == m) ? grad_out : 0
//
// Equality is exact here because m is literally one of the X[i] values (no
// arithmetic), so the comparison is well-defined. Ties (multiple elements equal
// to m) each receive the gradient — a standard subgradient choice for max.
//
// Buffers (indices auto-prepended to the push block, in this order):
//   input_idx     X            (full input shape)
//   max_idx       m            (scalar [1], the forward output)
//   grad_out_idx  grad scalar  (scalar [1], upstream gradient)
//   grad_in_idx   grad_in      (full input shape, output)

import Storage;

struct PushConstants {
	uint input_idx;
	uint max_idx;
	uint grad_out_idx;
	uint grad_in_idx;
	uint count;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint idx = tid.x;
	if (idx >= pc.count) return;

	float x = OaLoad(heap[pc.input_idx], idx);
	float m = OaLoad(heap[pc.max_idx], 0);
	float g = OaLoad(heap[pc.grad_out_idx], 0);

	OaStore(heap[pc.grad_in_idx], idx, (x == m) ? g : 0.0f);
}
