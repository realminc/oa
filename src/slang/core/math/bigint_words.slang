// Public-data fixed-width unsigned arithmetic, little-endian 32-bit limbs.
// No OA C++ BigInt donor exists (crypto schemas, source and tests audited).
// New implementation: radix carry/borrow, schoolbook product and binary long
// division identities. Reference: Handbook of Applied Cryptography, chapter 14.
// Not a constant-time or secret-device implementation. Operand-dependent
// comparison/division branches are intentional in this public-data baseline.
import wide_u32;

static const uint BIGINT_MAX_WORDS = 128u;

// Precondition for every helper: 1 <= words <= BIGINT_MAX_WORDS; inputs are
// initialized over the used extent. Unused output limbs are also initialized.
void bigint_zero(out uint result[128]) {
	for (uint i = 0u; i < BIGINT_MAX_WORDS; ++i) result[i] = 0u;
}

int bigint_compare(uint a[128], uint b[128], uint words) {
	for (uint i = words; i != 0u; --i) {
		if (a[i - 1u] < b[i - 1u]) return -1;
		if (a[i - 1u] > b[i - 1u]) return 1;
	}
	return 0;
}

uint bigint_add(uint a[128], uint b[128], uint words, out uint result[128]) {
	bigint_zero(result);
	uint carry = 0u;
	for (uint i = 0u; i < words; ++i) {
		uint sum = a[i] + b[i];
		uint carry_ab = uint(sum < a[i]);
		uint value = sum + carry;
		carry = carry_ab | uint(value < sum);
		result[i] = value;
	}
	return carry;
}

uint bigint_subtract(uint a[128], uint b[128], uint words, out uint result[128]) {
	bigint_zero(result);
	uint borrow = 0u;
	for (uint i = 0u; i < words; ++i) {
		uint difference = a[i] - b[i];
		uint borrow_ab = uint(a[i] < b[i]);
		uint value = difference - borrow;
		borrow = borrow_ab | uint(difference < borrow);
		result[i] = value;
	}
	return borrow;
}

void bigint_multiply(uint a[128], uint b[128], uint words, uint result_words, out uint result[256]) {
	for (uint i = 0u; i < 256u; ++i) result[i] = 0u;
	for (uint i = 0u; i < words; ++i) {
		uint carry = 0u;
		for (uint j = 0u; j < min(words, result_words - i); ++j) {
			uint k = i + j;
			uint2 product = mul32_wide(a[i], b[j]);
			uint sum = product.x + result[k];
			uint carry_product = uint(sum < product.x);
			uint value = sum + carry;
			// a[i]*b[j] + result[k] + carry <= 2^64 - 1.
			carry = product.y + carry_product + uint(value < sum);
			result[k] = value;
		}
		// Earlier outer rows end strictly below this fresh high limb.
		if (i + words < result_words) result[i + words] = carry;
	}
}

// Restoring binary division. On a zero divisor, status=false, quotient=0,
// remainder=dividend. Neither invalid content nor failure is hidden by a host
// readback. The caller propagates the status to its typed device output.
bool bigint_divide(
	uint dividend[128],
	uint divisor[128],
	uint words,
	out uint quotient[128],
	out uint remainder[128]
) {
	bigint_zero(quotient);
	bigint_zero(remainder);
	uint nonzero = 0u;
	for (uint i = 0u; i < words; ++i) nonzero |= divisor[i];
	if (nonzero == 0u) {
		for (uint i = 0u; i < words; ++i) remainder[i] = dividend[i];
		return false;
	}
	for (uint bit = words * 32u; bit != 0u; --bit) {
		uint index = bit - 1u;
		uint carry = (dividend[index / 32u] >> (index % 32u)) & 1u;
		for (uint i = 0u; i < words; ++i) {
			uint next = remainder[i] >> 31u;
			remainder[i] = (remainder[i] << 1u) | carry;
			carry = next;
		}
		// The shifted remainder may have one extra high bit. Keeping that
		// bit is necessary when the divisor's top limb has bit 31 set.
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

// Count over the declared fixed width, including high zero limbs. Zero has
// words*32 leading zeros; firstbithigh is never evaluated on zero.
uint bigint_leading_zeros(uint a[128], uint words) {
	for (uint i = words; i != 0u; --i) {
		if (a[i - 1u] != 0u)
			return (words - i) * 32u + 31u - uint(firstbithigh(a[i - 1u]));
	}
	return words * 32u;
}
