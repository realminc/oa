// Test-only entry points; import production formulas without copying them.
// Independent host oracles and Vulkan target validation are owned by Python.
import mlkem_kem;

StructuredBuffer<uint> inputs;
RWStructuredBuffer<uint> outputs;

[shader("compute")]
[numthreads(1, 1, 1)]
void mlkem_kem_components(uint3 tid : SV_DispatchThreadID) {
	uint base = tid.x * 73u, result = tid.x * 6369u;
	uint k = inputs[base], d[8], z[32], message[32], key[32], recovered[32];
	uint ek[1568], dk[3168], ciphertext[1568];
	for (uint i = 0u; i < 8u; ++i) d[i] = inputs[base + 1u + i];
	for (uint i = 0u; i < 32u; ++i) { z[i] = inputs[base + 9u + i]; message[i] = inputs[base + 41u + i]; }
	for (uint i = 0u; i < 1568u; ++i) { ek[i] = 0xa5u; ciphertext[i] = 0xa5u; }
	for (uint i = 0u; i < 3168u; ++i) dk[i] = 0xa5u;
	mlkem_keygen_internal(d, z, k, ek, dk);
	bool canonical = mlkem_encaps_internal(ek, message, k, key, ciphertext);
	mlkem_decaps_internal(dk, ciphertext, k, recovered);
	for (uint i = 0u; i < 1568u; ++i) { outputs[result + i] = ek[i]; outputs[result + 4736u + i] = ciphertext[i]; }
	for (uint i = 0u; i < 3168u; ++i) outputs[result + 1568u + i] = dk[i];
	for (uint i = 0u; i < 32u; ++i) { outputs[result + 6304u + i] = key[i]; outputs[result + 6336u + i] = recovered[i]; }
	outputs[result + 6368u] = canonical ? 1u : 0u;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void mlkem_kem_decaps_components(uint3 tid : SV_DispatchThreadID) {
	uint base = tid.x * 4737u, k = inputs[base], dk[3168], ciphertext[1568], key[32];
	for (uint i = 0u; i < 3168u; ++i) dk[i] = inputs[base + 1u + i];
	for (uint i = 0u; i < 1568u; ++i) ciphertext[i] = inputs[base + 3169u + i];
	mlkem_decaps_internal(dk, ciphertext, k, key);
	for (uint i = 0u; i < 32u; ++i) outputs[tid.x * 32u + i] = key[i];
}

[shader("compute")]
[numthreads(1, 1, 1)]
void mlkem_keygen_components(uint3 tid : SV_DispatchThreadID) {
	uint base = tid.x * 41u, k = inputs[base], d[8], z[32], ek[1568], dk[3168];
	for (uint i = 0u; i < 8u; ++i) d[i] = inputs[base + 1u + i];
	for (uint i = 0u; i < 32u; ++i) z[i] = inputs[base + 9u + i];
	for (uint i = 0u; i < 1568u; ++i) ek[i] = 0xa5u;
	for (uint i = 0u; i < 3168u; ++i) dk[i] = 0xa5u;
	mlkem_keygen_internal(d, z, k, ek, dk);
	for (uint i = 0u; i < 1568u; ++i) outputs[tid.x * 4736u + i] = ek[i];
	for (uint i = 0u; i < 3168u; ++i) outputs[tid.x * 4736u + 1568u + i] = dk[i];
}

[shader("compute")]
[numthreads(1, 1, 1)]
void mlkem_encaps_components(uint3 tid : SV_DispatchThreadID) {
	uint base = tid.x * 1601u, k = inputs[base], ek[1568], message[32], key[32], ciphertext[1568];
	for (uint i = 0u; i < 1568u; ++i) { ek[i] = inputs[base + 1u + i]; ciphertext[i] = 0xa5u; }
	for (uint i = 0u; i < 32u; ++i) message[i] = inputs[base + 1569u + i];
	mlkem_encaps_internal(ek, message, k, key, ciphertext);
	for (uint i = 0u; i < 1568u; ++i) outputs[tid.x * 1600u + i] = ciphertext[i];
	for (uint i = 0u; i < 32u; ++i) outputs[tid.x * 1600u + 1568u + i] = key[i];
}

[shader("compute")]
[numthreads(1, 1, 1)]
void mlkem_key_check_components(uint3 tid : SV_DispatchThreadID) {
	uint base = tid.x * 4740u, k = inputs[base], ek[1568], dk[3168];
	for (uint i = 0u; i < 1568u; ++i) ek[i] = inputs[base + 4u + i];
	for (uint i = 0u; i < 3168u; ++i) dk[i] = inputs[base + 1572u + i];
	outputs[tid.x * 3u] = mlkem_encapsulation_key_valid(ek, k, inputs[base + 1u]) ? 1u : 0u;
	outputs[tid.x * 3u + 1u] = mlkem_decapsulation_key_valid(dk, k, inputs[base + 2u]) ? 1u : 0u;
	outputs[tid.x * 3u + 2u] = mlkem_ciphertext_length_valid(k, inputs[base + 3u]) ? 1u : 0u;
}
