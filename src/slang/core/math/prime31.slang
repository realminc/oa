// Portable Montgomery arithmetic for canonical residues modulo an odd prime
// below 2^31. Reuses OA's independently proved exact 32x32 word product.
// R = 2^32; callers provide -p^-1 mod R. No shaderInt64 is required.
import wide_u32;

uint prime31_add(uint a, uint b, uint prime) {
	uint sum = a + b;
	return sum >= prime ? sum - prime : sum;
}

uint prime31_multiply(uint a, uint b, uint prime, uint negative_inverse) {
	uint2 product = mul32_wide(a, b);
	uint factor = product.x * negative_inverse;
	uint2 correction = mul32_wide(factor, prime);
	uint low = product.x + correction.x;
	uint high = product.y + correction.y + uint(low < product.x);
	return high >= prime ? high - prime : high;
}

uint prime31_power(uint a, uint exponent, uint prime, uint negative_inverse, uint one) {
	uint result = one;
	while (exponent != 0u) {
		if ((exponent & 1u) != 0u) result = prime31_multiply(result, a, prime, negative_inverse);
		a = prime31_multiply(a, a, prime, negative_inverse);
		exponent >>= 1u;
	}
	return result;
}
