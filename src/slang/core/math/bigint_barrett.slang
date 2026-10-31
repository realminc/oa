// Public-data Barrett reduction with a modulus-specific exact reciprocal.
// No OA C++ donor exists. Authority: Barrett radix reduction identities,
// Handbook of Applied Cryptography 14.3.3. No secret/constant-time claim.
import bigint_words;
import bigint_modular;
import wide_u32;

// Larger temporary operands are required for the reciprocal's extra limb.
// Preconditions: both lengths in 1..=130; output extent <= 260 words.
void bigint_barrett_multiply_limbs(
	uint a[130],
	uint a_words,
	uint b[130],
	uint b_words,
	out uint result[260]
) {
	for (uint i = 0u; i < 260u; ++i) result[i] = 0u;
	[loop]
	for (uint i = 0u; i < a_words; ++i) {
		uint carry = 0u;
		[loop]
		for (uint j = 0u; j < b_words; ++j) {
			uint2 product = mul32_wide(a[i], b[j]);
			uint sum = product.x + result[i + j];
			uint carry_product = uint(sum < product.x);
			uint value = sum + carry;
			carry = product.y + carry_product + uint(value < sum);
			result[i + j] = value;
		}
		result[i + b_words] = carry;
	}
}

// mu = floor(2^(64*k) / m). Significant modulus width k matters: leading zero
// limbs do not change the reduction radix. A radix-power modulus may require
// k+2 reciprocal limbs, including modulus one; never silently truncate it.
bool bigint_barrett_reciprocal(
	uint modulus[128],
	uint words,
	out uint mu[130],
	out uint significant_words
) {
	for (uint i = 0u; i < 130u; ++i) mu[i] = 0u;
	significant_words = 0u;
	for (uint i = 0u; i < words; ++i)
		if (modulus[i] != 0u) significant_words = i + 1u;
	if (significant_words == 0u) return false;
	uint remainder[128];
	bigint_zero(remainder);
	uint highest_bit = significant_words * 64u;
	[loop]
	for (uint bit = highest_bit + 1u; bit != 0u; --bit) {
		uint index = bit - 1u;
		uint carry = uint(index == highest_bit);
		for (uint i = 0u; i < significant_words; ++i) {
			uint next = remainder[i] >> 31u;
			remainder[i] = (remainder[i] << 1u) | carry;
			carry = next;
		}
		if (carry != 0u || bigint_compare(remainder, modulus, significant_words) >= 0) {
			uint borrow = 0u;
			for (uint i = 0u; i < significant_words; ++i) {
				uint difference = remainder[i] - modulus[i];
				uint borrow_ab = uint(remainder[i] < modulus[i]);
				uint value = difference - borrow;
				borrow = borrow_ab | uint(difference < borrow);
				remainder[i] = value;
			}
			// m >= radix^(k-1) proves mu <= radix^(k+1), fitting 130 limbs.
			mu[index / 32u] |= 1u << (index % 32u);
		}
	}
	return true;
}

// Preconditions: 1 <= k <= 128 is the significant modulus width, nonzero m,
// matching exact reciprocal, and input < 2^(64*k). Prepared data is public.
void bigint_barrett_reduce(
	uint input[256],
	uint modulus[128],
	uint mu[130],
	uint k,
	out uint result[128]
) {
	uint quotient_low[130], quotient_high[130], modulus_words[130];
	for (uint i = 0u; i < 130u; ++i) {
		quotient_low[i] = 0u; quotient_high[i] = 0u; modulus_words[i] = 0u;
	}
	for (uint i = 0u; i <= k; ++i) quotient_low[i] = input[i + k - 1u];
	for (uint i = 0u; i < k; ++i) modulus_words[i] = modulus[i];
	uint product[260];
	bigint_barrett_multiply_limbs(quotient_low, k + 1u, mu, k + 2u, product);
	for (uint i = 0u; i <= k; ++i) quotient_high[i] = product[i + k + 1u];
	bigint_barrett_multiply_limbs(quotient_high, k + 1u, modulus_words, k, product);
	uint low[128];
	bigint_zero(low);
	uint borrow = 0u;
	uint high = 0u;
	for (uint i = 0u; i <= k; ++i) {
		uint difference = input[i] - product[i];
		uint borrow_ab = uint(input[i] < product[i]);
		uint value = difference - borrow;
		borrow = borrow_ab | uint(difference < borrow);
		if (i < k) low[i] = value;
		else high = value;
	}
	// Subtraction wraps modulo radix^(k+1). The reciprocal bound proves the
	// wrapped remainder is < 3*m, so at most two corrections are necessary.
	[loop]
	while (high != 0u || bigint_compare(low, modulus, k) >= 0) {
		borrow = 0u;
		for (uint i = 0u; i < k; ++i) {
			uint difference = low[i] - modulus[i];
			uint borrow_ab = uint(low[i] < modulus[i]);
			uint value = difference - borrow;
			borrow = borrow_ab | uint(difference < borrow);
			low[i] = value;
		}
		high -= borrow;
	}
	for (uint i = 0u; i < 128u; ++i) result[i] = low[i];
}

bool bigint_barrett_multiply(
	uint a[128],
	uint b[128],
	uint modulus[128],
	uint words,
	out uint result[128]
) {
	bigint_zero(result);
	uint mu[130], k;
	if (!bigint_barrett_reciprocal(modulus, words, mu, k)) return false;
	uint left[128], right[128];
	bigint_reduce(a, modulus, words, left);
	bigint_reduce(b, modulus, words, right);
	uint product[256];
	bigint_multiply(left, right, k, k * 2u, product);
	bigint_barrett_reduce(product, modulus, mu, k, result);
	return true;
}
