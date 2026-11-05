// Public fixtures only. Output transports the signature and the actual cache;
// guards and poison detect workspace boundaries. No production secret admission.
import mldsa_sign;
StructuredBuffer<uint> inputs;
RWByteAddressBuffer outputs;

[shader("compute")]
[numthreads(1, 1, 1)]
void mldsa_sign_workspace_components(uint3 tid : SV_DispatchThreadID) {
	if (tid.x != 0u) return;
	uint key[4896], rnd[32], mu[64], signature[4627], attempts;
	for (uint i = 0u; i < 4896u; ++i) key[i] = inputs[2u + i];
	for (uint i = 0u; i < 32u; ++i) rnd[i] = inputs[4898u + i];
	for (uint i = 0u; i < 64u; ++i) mu[i] = inputs[4930u + i];
	for (uint i = 0u; i < 4627u; ++i) signature[i] = 0xa5u;
	bool valid = mldsa_sign_mu_buffer(inputs[1], key, mu, rnd,
		outputs, inputs[0], signature, attempts);
	for (uint i = 0u; i < 4627u; ++i) outputs.Store(i * 4u, signature[i]);
	outputs.Store(4627u * 4u, valid ? 1u : 0u);
	outputs.Store(4628u * 4u, attempts);
}
