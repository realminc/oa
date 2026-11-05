// Test-only transport for production FIPS 204 key generation.
// All seeds and expanded keys are public laboratory fixtures.
import mldsa_keygen;

StructuredBuffer<uint> inputs;
RWStructuredBuffer<uint> outputs;

[shader("compute")]
[numthreads(1, 1, 1)]
void mldsa_keygen_components(uint3 tid : SV_DispatchThreadID) {
	uint base = tid.x * 33u, result = tid.x * 7489u;
	uint seed[32], public_key[2592], secret_key[4896];
	for (uint i = 0u; i < 32u; ++i) seed[i] = inputs[base + 1u + i];
	for (uint i = 0u; i < 2592u; ++i) public_key[i] = 0xa5u;
	for (uint i = 0u; i < 4896u; ++i) secret_key[i] = 0xa5u;
	bool valid = mldsa_keygen_internal(inputs[base], seed, public_key, secret_key);
	for (uint i = 0u; i < 2592u; ++i) outputs[result + i] = public_key[i];
	for (uint i = 0u; i < 4896u; ++i) outputs[result + 2592u + i] = secret_key[i];
	outputs[result + 7488u] = valid ? 1u : 0u;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void mldsa_secret_sampling_components(uint3 tid : SV_DispatchThreadID) {
	uint base = tid.x * 66u, seed[64], poly[256];
	for (uint i = 0u; i < 64u; ++i) seed[i] = inputs[base + 2u + i];
	mldsa_expand_secret_poly(seed, inputs[base], inputs[base + 1u], poly);
	for (uint i = 0u; i < 256u; ++i) outputs[tid.x * 256u + i] = poly[i];
}
