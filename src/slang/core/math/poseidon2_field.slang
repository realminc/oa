// Adapted from Plonky3 poseidon2/src/external.rs and monty-31/src/poseidon2.rs
// at f98331687e47106a64ab82a77e2e1d0d66c9161b (MIT; see NOTICE.md).
// Mechanical field/array adaptation: Montgomery U32, fixed 24-word capacity.
// The permutation uses MDSMat4, not the distinct Horizon Labs 4x4 matrix.
import prime31;

void poseidon2_external(inout uint state[24], uint width, uint prime) {
	for (uint base = 0u; base < width; base += 4u) {
		uint x0 = state[base], x1 = state[base + 1u];
		uint x2 = state[base + 2u], x3 = state[base + 3u];
		uint t01 = prime31_add(x0, x1, prime);
		uint t23 = prime31_add(x2, x3, prime);
		uint sum = prime31_add(t01, t23, prime);
		uint with_x1 = prime31_add(sum, x1, prime);
		uint with_x3 = prime31_add(sum, x3, prime);
		state[base] = prime31_add(with_x1, t01, prime);
		state[base + 1u] = prime31_add(with_x1, prime31_add(x2, x2, prime), prime);
		state[base + 2u] = prime31_add(with_x3, t23, prime);
		state[base + 3u] = prime31_add(with_x3, prime31_add(x0, x0, prime), prime);
	}
	uint sums[4] = {0u, 0u, 0u, 0u};
	for (uint i = 0u; i < width; ++i) sums[i % 4u] = prime31_add(sums[i % 4u], state[i], prime);
	for (uint i = 0u; i < width; ++i) state[i] = prime31_add(state[i], sums[i % 4u], prime);
}

uint poseidon2_sbox(uint a, uint degree, uint prime, uint negative_inverse) {
	uint square = prime31_multiply(a, a, prime, negative_inverse);
	if (degree == 3u) return prime31_multiply(square, a, prime, negative_inverse);
	uint fourth = prime31_multiply(square, square, prime, negative_inverse);
	return prime31_multiply(prime31_multiply(fourth, square, prime, negative_inverse), a, prime, negative_inverse);
}

void poseidon2_internal(
	inout uint state[24],
	uint diagonal[24],
	uint width,
	uint prime,
	uint negative_inverse
) {
	uint sum = 0u;
	for (uint i = 0u; i < width; ++i) sum = prime31_add(sum, state[i], prime);
	for (uint i = 0u; i < width; ++i)
		state[i] = prime31_add(sum, prime31_multiply(state[i], diagonal[i], prime, negative_inverse), prime);
}
