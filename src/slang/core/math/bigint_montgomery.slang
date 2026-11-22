// Public-data Montgomery arithmetic, R = 2^(32*words), odd nonzero modulus.
// No OA C++ BigInt donor exists. New CIOS limb composition over the qualified
// exact word product; authority: Montgomery reduction identities, HAC 14.3.2.
// Canonical operand checks/normalization are public-data policy, not a secret
// execution or constant-time guarantee.
import bigint_words;
import wide_u32;

uint bigint_montgomery_factor(uint low_modulus) {
	// Starting with one is correct modulo two for every odd low word. Each
	// Newton step doubles the number of correct inverse bits, up to 32.
	uint inverse = 1u;
	for (uint i = 0u; i < 5u; ++i) inverse *= 2u - low_modulus * inverse;
	return 0u - inverse;
}

// Preconditions: odd m != 0, canonical a,b < m; negative_inverse = -m[0]^-1
// modulo 2^32. The extra two accumulator limbs retain both addition carries.
void bigint_montgomery_product(
	uint a[128],
	uint b[128],
	uint modulus[128],
	uint words,
	uint negative_inverse,
	out uint result[128]
) {
	uint accumulator[130];
	for (uint i = 0u; i < 130u; ++i) accumulator[i] = 0u;
	[loop]
	for (uint i = 0u; i < words; ++i) {
		uint carry = 0u;
		[loop]
		for (uint j = 0u; j < words; ++j) {
			uint2 product = mul32_wide(a[j], b[i]);
			uint sum = product.x + accumulator[j];
			uint carry_product = uint(sum < product.x);
			uint value = sum + carry;
			carry = product.y + carry_product + uint(value < sum);
			accumulator[j] = value;
		}
		uint high_sum = accumulator[words] + carry;
		accumulator[words + 1u] += uint(high_sum < accumulator[words]);
		accumulator[words] = high_sum;

		uint factor = accumulator[0] * negative_inverse;
		carry = 0u;
		[loop]
		for (uint j = 0u; j < words; ++j) {
			uint2 product = mul32_wide(factor, modulus[j]);
			uint sum = product.x + accumulator[j];
			uint carry_product = uint(sum < product.x);
			uint value = sum + carry;
			carry = product.y + carry_product + uint(value < sum);
			accumulator[j] = value;
		}
		high_sum = accumulator[words] + carry;
		accumulator[words + 1u] += uint(high_sum < accumulator[words]);
		accumulator[words] = high_sum;
		// The exact zero low limb can now be discarded (division by 2^32).
		for (uint j = 0u; j <= words; ++j) accumulator[j] = accumulator[j + 1u];
		accumulator[words + 1u] = 0u;
	}
	uint low[128];
	bigint_zero(low);
	for (uint i = 0u; i < words; ++i) low[i] = accumulator[i];
	if (accumulator[words] != 0u || bigint_compare(low, modulus, words) >= 0)
		bigint_subtract(low, modulus, words, result);
	else for (uint i = 0u; i < 128u; ++i) result[i] = low[i];
}
