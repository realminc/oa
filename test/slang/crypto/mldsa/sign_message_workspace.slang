// One variable-sized public fixture; packed-byte input and guarded uint output.
import mldsa_sign;
import mldsa_packing;
RWByteAddressBuffer inputs;
RWByteAddressBuffer outputs;

[shader("compute")]
[numthreads(1, 1, 1)]
void mldsa_sign_message_workspace_components(uint3 tid : SV_DispatchThreadID) {
	if (tid.x != 0u) return;
	uint key[4896], rnd[32], signature[4627], attempts;
	// Harness preflights a complete fixed header and key/rnd transport region.
	for (uint i = 0u; i < 4896u; ++i) key[i] = read_byte(inputs, 60u + i);
	for (uint i = 0u; i < 32u; ++i) rnd[i] = read_byte(inputs, 28u + i);
	for (uint i = 0u; i < 4627u; ++i) signature[i] = 0xa5u;
	bool valid = mldsa_sign_message_buffer(inputs.Load(0u), key, rnd,
		inputs, inputs.Load(4u), inputs.Load(8u),
		inputs, inputs.Load(12u), inputs.Load(16u), inputs.Load(20u) != 0u,
		outputs, inputs.Load(24u), signature, attempts);
	for (uint i = 0u; i < 4627u; ++i) outputs.Store(i * 4u, signature[i]);
	outputs.Store(4627u * 4u, valid ? 1u : 0u);
	outputs.Store(4628u * 4u, attempts);
}
