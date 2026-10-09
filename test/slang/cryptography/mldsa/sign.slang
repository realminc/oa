// Fixed laboratory transport. Expanded keys and entropy are public NIST fixtures.
import mldsa_sign;
StructuredBuffer<uint> inputs;
RWStructuredBuffer<uint> outputs;

[shader("compute")]
[numthreads(1, 1, 1)]
void mldsa_sign_components(uint3 tid : SV_DispatchThreadID) {
	uint base = tid.x * 4993u, result = tid.x * 4629u;
	uint key[4896], rnd[32], mu[64], signature[4627], attempts;
	for (uint i = 0u; i < 4896u; ++i) key[i] = inputs[base + 1u + i];
	for (uint i = 0u; i < 32u; ++i) rnd[i] = inputs[base + 4897u + i];
	for (uint i = 0u; i < 64u; ++i) mu[i] = inputs[base + 4929u + i];
	for (uint i = 0u; i < 4627u; ++i) signature[i] = 0xa5u;
	bool valid = mldsa_sign_mu(inputs[base], key, mu, rnd, signature, attempts);
	for (uint i = 0u; i < 4627u; ++i) outputs[result + i] = signature[i];
	outputs[result + 4627u] = valid ? 1u : 0u;
	outputs[result + 4628u] = attempts;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void mldsa_mask_components(uint3 tid : SV_DispatchThreadID) {
	uint base = tid.x * 66u, seed[64], poly[256];
	for (uint i = 0u; i < 64u; ++i) seed[i] = inputs[base + 2u + i];
	bool small = inputs[base + 1u] == 44u;
	mldsa_expand_mask_poly(seed, inputs[base], small ? 18u : 20u, small ? 131072u : 524288u, poly);
	for (uint i = 0u; i < 256u; ++i) outputs[tid.x * 256u + i] = poly[i];
}
