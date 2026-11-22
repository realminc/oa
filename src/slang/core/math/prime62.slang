// Portable public-field arithmetic for odd moduli below 2^62, encoded uint2
// (low word first). Montgomery R = 2^64. No OA C++ generic NTT donor exists.
// CIOS composition over OA's qualified 32x32 product, HAC chapter 14.
import wide_u32;

bool prime62_ge(uint2 a, uint2 b) {
	return a.y > b.y || (a.y == b.y && a.x >= b.x);
}

uint2 prime62_subtract(uint2 a, uint2 b) {
	return uint2(a.x - b.x, a.y - b.y - uint(a.x < b.x));
}

uint2 prime62_add(uint2 a, uint2 b, uint2 modulus) {
	uint low = a.x + b.x;
	uint2 result = uint2(low, a.y + b.y + uint(low < a.x));
	return prime62_ge(result, modulus) ? prime62_subtract(result, modulus) : result;
}

uint2 prime62_difference(uint2 a, uint2 b, uint2 modulus) {
	if (prime62_ge(a, b)) return prime62_subtract(a, b);
	return prime62_subtract(modulus, prime62_subtract(b, a));
}

uint2 prime62_normalize(uint2 input, uint2 modulus) {
	if (!prime62_ge(input, modulus)) return input;
	uint2 result = uint2(0u, 0u);
	for (uint bit = 64u; bit != 0u; --bit) {
		uint index = bit - 1u;
		uint value = index < 32u ? ((input.x >> index) & 1u) : ((input.y >> (index - 32u)) & 1u);
		result = prime62_add(result, result, modulus);
		result = prime62_add(result, uint2(value, 0u), modulus);
	}
	return result;
}

uint2 prime62_multiply(uint2 a, uint2 b, uint2 modulus, uint factor) {
	uint accumulator[4] = {0u, 0u, 0u, 0u};
	for (uint i = 0u; i < 2u; ++i) {
		uint carry = 0u;
		for (uint j = 0u; j < 2u; ++j) {
			uint2 product = mul32_wide(a[j], b[i]);
			uint sum = product.x + accumulator[j];
			uint value = sum + carry;
			carry = product.y + uint(sum < product.x) + uint(value < sum);
			accumulator[j] = value;
		}
		uint sum = accumulator[2] + carry;
		accumulator[3] += uint(sum < accumulator[2]); accumulator[2] = sum;
		uint multiplier = accumulator[0] * factor;
		carry = 0u;
		for (uint j = 0u; j < 2u; ++j) {
			uint2 product = mul32_wide(multiplier, modulus[j]);
			sum = product.x + accumulator[j];
			uint value = sum + carry;
			carry = product.y + uint(sum < product.x) + uint(value < sum);
			accumulator[j] = value;
		}
		sum = accumulator[2] + carry;
		accumulator[3] += uint(sum < accumulator[2]); accumulator[2] = sum;
		for (uint j = 0u; j < 3u; ++j) accumulator[j] = accumulator[j + 1u];
		accumulator[3] = 0u;
	}
	uint2 result = uint2(accumulator[0], accumulator[1]);
	return prime62_ge(result, modulus) ? prime62_subtract(result, modulus) : result;
}

uint2 prime62_power(uint2 a, uint exponent, uint2 modulus, uint factor, uint2 one) {
	uint2 result = one;
	while (exponent != 0u) {
		if ((exponent & 1u) != 0u) result = prime62_multiply(result, a, modulus, factor);
		a = prime62_multiply(a, a, modulus, factor);
		exponent >>= 1u;
	}
	return result;
}
