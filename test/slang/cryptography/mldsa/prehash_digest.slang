// Test-only transport; schema-owned selectors and production formulas are imported.
import mldsa_prehash_digest;
RWByteAddressBuffer inputs;
RWStructuredBuffer<uint> outputs;

[numthreads(1, 1, 1)]
void mldsa_prehash_digest_components(uint3 id : SV_DispatchThreadID) {
	if (id.x != 0u) return;
	uint algorithm = inputs.Load(0u);
	uint base = inputs.Load(4u);
	uint length = inputs.Load(8u);
	uint tr_base = inputs.Load(12u);
	uint context_base = inputs.Load(16u);
	uint context_length = inputs.Load(20u);
	uint digest[16], mu[16];
	for (uint i = 0u; i < 16u; ++i) {
		digest[i] = 0xa5a5a5a5u;
		mu[i] = 0xa5a5a5a5u;
	}
	outputs[0] = mldsa_prehash_digest(inputs, base, length, algorithm, digest) ? 1u : 0u;
	outputs[1] = mldsa_prehash_mu(inputs, base, length, inputs, tr_base,
		inputs, context_base, context_length, algorithm, mu) ? 1u : 0u;
	for (uint i = 0u; i < 16u; ++i) {
		outputs[2u + i] = digest[i];
		outputs[18u + i] = mu[i];
	}
}
