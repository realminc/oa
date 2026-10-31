// Test-only public fixtures around production KEM buffer storage composition.
// Guards and poisoned tails belong to the host harness; no secret API admission.
import mlkem_kem;
import mlkem_kpke;

StructuredBuffer<uint> inputs;
RWByteAddressBuffer outputs;

[shader("compute")]
[numthreads(1, 1, 1)]
void mlkem_kem_workspace_components(uint3 tid : SV_DispatchThreadID) {
	uint input_base = tid.x * 74u;
	uint result = tid.x * 7398u;
	uint k = inputs[input_base], override_base = inputs[input_base + 1u];
	uint base = override_base == 0u ? (result + 6373u) * 4u : override_base;
	uint d[8], z[32], message[32], key[32], recovered[32];
	uint ek[1568], dk[3168], ciphertext[1568];
	for (uint i = 0u; i < 8u; ++i) d[i] = inputs[input_base + 2u + i];
	for (uint i = 0u; i < 32u; ++i) {
		z[i] = inputs[input_base + 10u + i];
		message[i] = inputs[input_base + 42u + i];
		key[i] = 0xa5u;
		recovered[i] = 0xa5u;
	}
	for (uint i = 0u; i < 1568u; ++i) {
		ek[i] = 0xa5u;
		ciphertext[i] = 0xa5u;
	}
	for (uint i = 0u; i < 3168u; ++i) dk[i] = 0xa5u;
	bool generated = mlkem_keygen_buffer(d, z, k, outputs, base, ek, dk);
	bool canonical = mlkem_encaps_buffer(ek, message, k, outputs, base, key, ciphertext);
	bool decrypted = mlkem_decaps_buffer(dk, ciphertext, k, outputs, base, recovered);
	for (uint i = 0u; i < 1568u; ++i) {
		outputs.Store((result + i) * 4u, ek[i]);
		outputs.Store((result + 4736u + i) * 4u, ciphertext[i]);
	}
	for (uint i = 0u; i < 3168u; ++i) outputs.Store((result + 1568u + i) * 4u, dk[i]);
	for (uint i = 0u; i < 32u; ++i) {
		outputs.Store((result + 6304u + i) * 4u, key[i]);
		outputs.Store((result + 6336u + i) * 4u, recovered[i]);
	}
	outputs.Store((result + 6368u) * 4u, uint(canonical));
	outputs.Store((result + 6369u) * 4u, uint(generated));
	outputs.Store((result + 6370u) * 4u, uint(decrypted));
	outputs.Store((result + 6371u) * 4u, uint(mlkem_vector_range_valid(outputs, base, k)));
}

[shader("compute")]
[numthreads(1, 1, 1)]
void mlkem_decaps_workspace_components(uint3 tid : SV_DispatchThreadID) {
	uint input_base = tid.x * 4738u;
	uint result = tid.x * 1059u;
	uint k = inputs[input_base], override_base = inputs[input_base + 1u];
	uint base = override_base == 0u ? (result + 34u) * 4u : override_base;
	uint dk[3168], ciphertext[1568], key[32];
	for (uint i = 0u; i < 3168u; ++i) dk[i] = inputs[input_base + 2u + i];
	for (uint i = 0u; i < 1568u; ++i) ciphertext[i] = inputs[input_base + 3170u + i];
	for (uint i = 0u; i < 32u; ++i) key[i] = 0xa5u;
	bool usable = mlkem_decaps_buffer(dk, ciphertext, k, outputs, base, key);
	for (uint i = 0u; i < 32u; ++i) outputs.Store((result + i) * 4u, key[i]);
	outputs.Store((result + 32u) * 4u, uint(usable));
}
