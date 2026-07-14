// SiluAndMulBwd — backward of SiluAndMul (Out = SiLU(gate) * up).
//
// Mirrors the flat split in SiluAndMul.slang: the input x is laid out as
// [gate(half) ; up(half)] with half = count/2, and y[i] = SiLU(x[i]) * x[i+half].
// The gradient needs the INPUT (gate and up), not the forward output — silu(g)*u
// cannot be inverted to recover g and u. With g=x[i], u=x[i+half], dy=dY[i]:
//   dX[i]        = dy * u * silu'(g)      (gate side), silu'(g) = sig*(1 + g*(1-sig))
//   dX[i+half]   = dy * silu(g)           (up side),   silu(g)  = g*sig
// where sig = sigmoid(g).

import Storage;
import Activations;  // Sigmoid — shared module (was inline; byte-identical)

struct PushConstants {
	uint x_idx;        // input gate||up  [count]
	uint dy_idx;       // upstream grad   (first half used)
	uint dx_idx;       // output dInput   [count]
	uint count;        // total input element count (= 2*half)
	uint intermediate_size;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
	uint i = gid.x * 256 + lid.x;
	uint outputCount = pc.count / 2;
	if (i >= outputCount) return;
	uint row = i / pc.intermediate_size;
	uint col = i % pc.intermediate_size;
	uint rowBase = row * (2 * pc.intermediate_size);
	uint gateIndex = rowBase + col;
	uint upIndex = rowBase + pc.intermediate_size + col;
	float g  = OaLoad(heap[pc.x_idx], gateIndex);
	float u  = OaLoad(heap[pc.x_idx], upIndex);
	float dy = OaLoad(heap[pc.dy_idx], i);

	float sig   = Sigmoid(g);
	float silu  = g * sig;
	float siluP = sig * (1.0f + g * (1.0f - sig));

	OaStore(heap[pc.dx_idx], gateIndex, dy * u * siluP);
	OaStore(heap[pc.dx_idx], upIndex, dy * silu);
}
