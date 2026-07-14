// GegluBwd — backward of Geglu (Out = up * GELU(gate)).
//
// Mirrors the flat split in Geglu.slang: input x = [up(half) ; gate(half)] with
// half = count/2, and y[i] = x[i] * GELU(x[i+half]). The gradient needs the INPUT
// (up and gate), not the forward output. With u=x[i], g=x[i+half], dy=dY[i]:
//   dX[i]      = dy * GELU(g)            (up side)
//   dX[i+half] = dy * u * GELU'(g)       (gate side)
// GELU and GELU' use the tanh approximation, matching Activations::Gelu / GeluBwd.

import Activations;
import Storage;

struct PushConstants {
	uint x_idx;        // input up||gate  [count]
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
	uint upIndex = rowBase + col;
	uint gateIndex = rowBase + pc.intermediate_size + col;
	float u  = OaLoad(heap[pc.x_idx], upIndex);
	float g  = OaLoad(heap[pc.x_idx], gateIndex);
	float dy = OaLoad(heap[pc.dy_idx], i);

	// GELU(g) and its derivative (tanh approximation, matches GeluBwd.slang).
	const float kSqrt2OverPi = 0.7978845608f;
	const float kCoeff = 0.044715f;
	float inner    = kSqrt2OverPi * (g + kCoeff * g * g * g);
	float tanhIn   = tanh(inner);
	float sech2    = 1.0f - tanhIn * tanhIn;
	float dInner   = kSqrt2OverPi * (1.0f + 3.0f * kCoeff * g * g);
	float geluG    = g * 0.5f * (1.0f + tanhIn);
	float geluP    = 0.5f * (1.0f + tanhIn) + 0.5f * g * sech2 * dInner;

	OaStore(heap[pc.dx_idx], upIndex, dy * geluG);
	OaStore(heap[pc.dx_idx], gateIndex, dy * u * geluP);
}
