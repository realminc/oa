// Test-only entry points; import production formulas without copying them.
// Independent host oracles and Vulkan target validation are owned by Python.
import mlkem_kpke;

StructuredBuffer<uint> inputs;
RWStructuredBuffer<uint> outputs;

[shader("compute")]
[numthreads(1, 1, 1)]
void mlkem_kpke_components(uint3 tid : SV_DispatchThreadID) {
	uint base = tid.x * 49u, result = tid.x * 4705u;
	uint k = inputs[base], d[8], r[8], message[32], recovered[32];
	uint ek[1568], dk[1536], ciphertext[1568];
	for (uint i = 0u; i < 8u; ++i) { d[i] = inputs[base + 1u + i]; r[i] = inputs[base + 9u + i]; }
	for (uint i = 0u; i < 32u; ++i) message[i] = inputs[base + 17u + i];
	for (uint i = 0u; i < 1568u; ++i) { ek[i] = 0xa5u; ciphertext[i] = 0xa5u; }
	for (uint i = 0u; i < 1536u; ++i) dk[i] = 0xa5u;
	mlkem_kpke_keygen(d, k, ek, dk);
	bool canonical = mlkem_kpke_encrypt(ek, message, r, k, ciphertext);
	mlkem_kpke_decrypt(dk, ciphertext, k, recovered);
	for (uint i = 0u; i < 1568u; ++i) { outputs[result + i] = ek[i]; outputs[result + 3104u + i] = ciphertext[i]; }
	for (uint i = 0u; i < 1536u; ++i) outputs[result + 1568u + i] = dk[i];
	for (uint i = 0u; i < 32u; ++i) outputs[result + 4672u + i] = recovered[i];
	outputs[result + 4704u] = canonical ? 1u : 0u;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void mlkem_kpke_decrypt_components(uint3 tid : SV_DispatchThreadID) {
	uint base = tid.x * 3105u, k = inputs[base];
	uint dk[1536], ciphertext[1568], message[32];
	for (uint i = 0u; i < 1536u; ++i) dk[i] = inputs[base + 1u + i];
	for (uint i = 0u; i < 1568u; ++i) ciphertext[i] = inputs[base + 1537u + i];
	mlkem_kpke_decrypt(dk, ciphertext, k, message);
	for (uint i = 0u; i < 32u; ++i) outputs[tid.x * 32u + i] = message[i];
}
