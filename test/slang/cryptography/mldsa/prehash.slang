// Public NIST fixture transport only; never a production secret executor.
import mldsa_prehash;
import mldsa_packing;
RWByteAddressBuffer inputs;
RWStructuredBuffer<uint> outputs;

[shader("compute")]
[numthreads(1, 1, 1)]
void mldsa_prehash_components(uint3 tid : SV_DispatchThreadID) {
	if (tid.x != 0u) return;
	uint key[4896], rnd[32], signature[4627], attempts;
	for (uint i = 0u; i < 4896u; ++i) key[i] = read_byte(inputs, 60u + i);
	for (uint i = 0u; i < 32u; ++i) rnd[i] = read_byte(inputs, 28u + i);
	for (uint i = 0u; i < 4627u; ++i) signature[i] = 0xa5u;
	bool valid = mldsa_sign_prehashed(inputs.Load(0u), key, rnd,
		inputs, inputs.Load(4u), inputs.Load(8u),
		inputs, inputs.Load(12u), inputs.Load(16u), inputs.Load(20u),
		signature, attempts);
	for (uint i = 0u; i < 4627u; ++i) outputs[i] = signature[i];
	outputs[4627] = valid ? 1u : 0u;
	outputs[4628] = attempts;
}
