// Test-only transport around the production buffered K-PKE formulas.
// Disposable public fixtures; no Engine or secret-operation admission.
import mlkem_kpke;

StructuredBuffer<uint> inputs;
RWByteAddressBuffer outputs;

[shader("compute")]
[numthreads(1, 1, 1)]
void mlkem_kpke_workspace_components(uint3 tid : SV_DispatchThreadID) {
	uint input_base = tid.x * 50u;
	uint result = tid.x * 5734u;
	uint k = inputs[input_base];
	uint override_base = inputs[input_base + 1u];
	uint base = override_base == 0u ? (result + 4709u) * 4u : override_base;
	uint d[8], r[8], message[32], recovered[32];
	uint ek[1568], dk[1536], ciphertext[1568];
	for (uint i = 0u; i < 8u; ++i) {
		d[i] = inputs[input_base + 2u + i];
		r[i] = inputs[input_base + 10u + i];
	}
	for (uint i = 0u; i < 32u; ++i) {
		message[i] = inputs[input_base + 18u + i];
		recovered[i] = 0xa5u;
	}
	for (uint i = 0u; i < 1568u; ++i) {
		ek[i] = 0xa5u;
		ciphertext[i] = 0xa5u;
	}
	for (uint i = 0u; i < 1536u; ++i) dk[i] = 0xa5u;
	bool generated = mlkem_kpke_keygen_buffer(d, k, outputs, base, ek, dk);
	bool canonical = mlkem_kpke_encrypt_buffer(ek, message, r, k, outputs, base, ciphertext);
	bool decrypted = mlkem_kpke_decrypt_buffer(dk, ciphertext, k, outputs, base, recovered);
	for (uint i = 0u; i < 1568u; ++i) {
		outputs.Store((result + i) * 4u, ek[i]);
		outputs.Store((result + 3104u + i) * 4u, ciphertext[i]);
	}
	for (uint i = 0u; i < 1536u; ++i) outputs.Store((result + 1568u + i) * 4u, dk[i]);
	for (uint i = 0u; i < 32u; ++i) outputs.Store((result + 4672u + i) * 4u, recovered[i]);
	outputs.Store((result + 4704u) * 4u, uint(canonical));
	outputs.Store((result + 4705u) * 4u, uint(generated));
	outputs.Store((result + 4706u) * 4u, uint(decrypted));
	outputs.Store((result + 4707u) * 4u, uint(mlkem_vector_range_valid(outputs, base, k)));
}
