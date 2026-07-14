// CausalMaskBwd — pass gradients on/below the diagonal and zero above it.

import Storage;

struct PushConstants {
	uint grad_idx;
	uint output_idx;
	uint count;
	uint Tq;
	uint Tk;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint i = tid.x;
	if (i >= pc.count) return;
	uint local = i % (pc.Tq * pc.Tk);
	uint q = local / pc.Tk;
	uint k = local % pc.Tk;
	OaStore(heap[pc.output_idx], i,
		k > q ? 0.0f : OaLoad(heap[pc.grad_idx], i));
}
