// Public-data mixed-width division. No OA C++ BigInt donor exists.
// Mechanical extension of the qualified restoring-division body: the dividend
// and quotient have dividend_words limbs; the remainder keeps words plus its carry bit.
// Preconditions: 1 <= words <= 128, 1 <= dividend_words <= 256. Zero divisor yields zero outputs/false.
import bigint_words;
import bigint_modular;
import bigint_montgomery;
import wide_u32;

bool bigint_divide_mixed(
	uint dividend[256],
	uint divisor[128],
	uint dividend_words,
	uint words,
	out uint quotient[256],
	out uint remainder[128]
) {
	for (uint i = 0u; i < 256u; ++i) quotient[i] = 0u;
	bigint_zero(remainder);
	uint nonzero = 0u;
	for (uint i = 0u; i < words; ++i) nonzero |= divisor[i];
	if (nonzero == 0u) return false;
	for (uint bit = dividend_words * 32u; bit != 0u; --bit) {
		uint index = bit - 1u;
		uint carry = (dividend[index / 32u] >> (index % 32u)) & 1u;
		for (uint i = 0u; i < words; ++i) {
			uint next = remainder[i] >> 31u;
			remainder[i] = (remainder[i] << 1u) | carry;
			carry = next;
		}
		if (carry != 0u || bigint_compare(remainder, divisor, words) >= 0) {
			uint borrow = 0u;
			for (uint i = 0u; i < words; ++i) {
				uint difference = remainder[i] - divisor[i];
				uint borrow_ab = uint(remainder[i] < divisor[i]);
				uint value = difference - borrow;
				borrow = borrow_ab | uint(difference < borrow);
				remainder[i] = value;
			}
			quotient[index / 32u] |= 1u << (index % 32u);
		}
	}
	return true;
}

// REDC for a full double-width input. Public inputs need not satisfy T < m*R.
// Keep the extra accumulator carry, then canonically reduce the N+1-word
// candidate. For the customary bounded input, at most one subtraction is needed.
bool bigint_montgomery_reduce_wide(
	uint input[256],
	uint modulus[128],
	uint words,
	out uint result[128]
) {
	bigint_zero(result);
	if ((modulus[0] & 1u) == 0u) return false;
	uint accumulator[257];
	for (uint i = 0u; i < 257u; ++i) accumulator[i] = 0u;
	for (uint i = 0u; i < words * 2u; ++i) accumulator[i] = input[i];
	uint negative_inverse = bigint_montgomery_factor(modulus[0]);
	for (uint i = 0u; i < words; ++i) {
		uint factor = accumulator[i] * negative_inverse;
		uint carry = 0u;
		for (uint j = 0u; j < words; ++j) {
			uint2 product = mul32_wide(factor, modulus[j]);
			uint sum = product.x + accumulator[i + j];
			uint carry_product = uint(sum < product.x);
			uint value = sum + carry;
			carry = product.y + carry_product + uint(value < sum);
			accumulator[i + j] = value;
		}
		uint position = i + words;
		while (carry != 0u && position <= words * 2u) {
			uint previous = accumulator[position];
			accumulator[position] = previous + carry;
			carry = uint(accumulator[position] < previous);
			++position;
		}
	}
	uint candidate[256];
	for (uint i = 0u; i < 256u; ++i) candidate[i] = 0u;
	for (uint i = 0u; i <= words; ++i) candidate[i] = accumulator[i + words];
	return bigint_reduce_wide(candidate, words + 1u, modulus, words, result);
}
