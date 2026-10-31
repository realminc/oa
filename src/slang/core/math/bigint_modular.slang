// Public-data modular arithmetic over little-endian 32-bit limbs.
// No qualified OA C++ BigInt donor exists. New implementation from integer
// remainder, Euclidean coefficient recurrence and binary exponentiation
// identities; see Handbook of Applied Cryptography, chapter 14.
// No constant-time, secret-placement or competitive-performance claim.
import bigint_words;
import bigint_montgomery;

bool bigint_is_zero(
	uint value[128],
	uint words
) {
	uint combined = 0u;
	for (uint i = 0u; i < words; ++i) combined |= value[i];
	return combined == 0u;
}

// Precondition: 1 <= words <= 128, 0 <= input_words <= 256. Ordinary residues.
// A zero modulus returns false and a completely zero result.
bool bigint_reduce_wide(
	uint input[256],
	uint input_words,
	uint modulus[128],
	uint words,
	out uint result[128]
) {
	bigint_zero(result);
	if (bigint_is_zero(modulus, words)) return false;
	[loop]
	for (uint bit = input_words * 32u; bit != 0u; --bit) {
		uint index = bit - 1u;
		uint carry = (input[index / 32u] >> (index % 32u)) & 1u;
		[loop]
		for (uint i = 0u; i < words; ++i) {
			uint next = result[i] >> 31u;
			result[i] = (result[i] << 1u) | carry;
			carry = next;
		}
		if (carry != 0u || bigint_compare(result, modulus, words) >= 0) {
			uint borrow = 0u;
			[loop]
			for (uint i = 0u; i < words; ++i) {
				uint difference = result[i] - modulus[i];
				uint borrow_ab = uint(result[i] < modulus[i]);
				uint value = difference - borrow;
				borrow = borrow_ab | uint(difference < borrow);
				result[i] = value;
			}
		}
	}
	return true;
}

bool bigint_reduce(
	uint input[128],
	uint modulus[128],
	uint words,
	out uint result[128]
) {
	uint quotient[128];
	bool valid = bigint_divide(input, modulus, words, quotient, result);
	if (!valid) bigint_zero(result);
	return valid;
}

bool bigint_add_mod(
	uint a[128],
	uint b[128],
	uint modulus[128],
	uint words,
	out uint result[128]
) {
	uint sum[128];
	uint carry = bigint_add(a, b, words, sum);
	uint wide[256];
	for (uint i = 0u; i < 256u; ++i) wide[i] = 0u;
	for (uint i = 0u; i < words; ++i) wide[i] = sum[i];
	wide[words] = carry;
	return bigint_reduce_wide(wide, words + 1u, modulus, words, result);
}

bool bigint_subtract_mod(
	uint a[128],
	uint b[128],
	uint modulus[128],
	uint words,
	out uint result[128]
) {
	bigint_zero(result);
	uint left[128], right[128], difference[128];
	if (!bigint_reduce(a, modulus, words, left)) return false;
	bigint_reduce(b, modulus, words, right);
	uint borrow = bigint_subtract(left, right, words, difference);
	if (borrow != 0u) bigint_add(difference, modulus, words, result);
	else for (uint i = 0u; i < 128u; ++i) result[i] = difference[i];
	return true;
}

bool bigint_multiply_mod(
	uint a[128],
	uint b[128],
	uint modulus[128],
	uint words,
	out uint result[128]
) {
	uint product[256];
	bigint_multiply(a, b, words, words * 2u, product);
	return bigint_reduce_wide(product, words * 2u, modulus, words, result);
}

// Euclidean coefficients are stored modulo m, avoiding signed fixed-width
// temporaries. This accepts odd/even and composite moduli; non-units return
// false and zero. The modulus-one ring follows Python's inverse convention.
bool bigint_inverse_mod(
	uint input[128],
	uint modulus[128],
	uint words,
	out uint result[128]
) {
	bigint_zero(result);
	uint old_r[128], r[128], old_t[128], t[128];
	if (!bigint_reduce(input, modulus, words, r)) return false;
	for (uint i = 0u; i < 128u; ++i) old_r[i] = modulus[i];
	bigint_zero(old_t);
	uint one[128];
	bigint_zero(one);
	one[0] = 1u;
	bigint_reduce(one, modulus, words, t);
	[loop]
	while (!bigint_is_zero(r, words)) {
		uint quotient[128], next_r[128], product[128], next_t[128];
		bigint_divide(old_r, r, words, quotient, next_r);
		bigint_multiply_mod(quotient, t, modulus, words, product);
		bigint_subtract_mod(old_t, product, modulus, words, next_t);
		for (uint i = 0u; i < 128u; ++i) {
			old_r[i] = r[i]; r[i] = next_r[i];
			old_t[i] = t[i]; t[i] = next_t[i];
		}
	}
	if (bigint_compare(old_r, one, words) != 0) return false;
	for (uint i = 0u; i < 128u; ++i) result[i] = old_t[i];
	return true;
}

bool bigint_montgomery_multiply(
	uint a[128],
	uint b[128],
	uint modulus[128],
	uint words,
	out uint result[128]
) {
	bigint_zero(result);
	if ((modulus[0] & 1u) == 0u) return false;
	uint left[128], right[128];
	bigint_reduce(a, modulus, words, left);
	bigint_reduce(b, modulus, words, right);
	bigint_montgomery_product(left, right, modulus, words, bigint_montgomery_factor(modulus[0]), result);
	return true;
}

bool bigint_to_montgomery(
	uint input[128],
	uint modulus[128],
	uint words,
	out uint result[128]
) {
	bigint_zero(result);
	if ((modulus[0] & 1u) == 0u) return false;
	uint shifted[256];
	for (uint i = 0u; i < 256u; ++i) shifted[i] = 0u;
	for (uint i = 0u; i < words; ++i) shifted[i + words] = input[i];
	return bigint_reduce_wide(shifted, words * 2u, modulus, words, result);
}

bool bigint_from_montgomery(
	uint input[128],
	uint modulus[128],
	uint words,
	out uint result[128]
) {
	uint one[128];
	bigint_zero(one); one[0] = 1u;
	return bigint_montgomery_multiply(input, one, modulus, words, result);
}

bool bigint_pow_mod(
	uint base[128],
	uint exponent[128],
	uint modulus[128],
	uint words,
	out uint result[128]
) {
	bigint_zero(result);
	uint power[128], accumulator[128], one[128];
	if (!bigint_reduce(base, modulus, words, power)) return false;
	bigint_zero(one); one[0] = 1u;
	bigint_reduce(one, modulus, words, accumulator);
	bool montgomery = (modulus[0] & 1u) != 0u;
	uint negative_inverse = 0u;
	if (montgomery) {
		uint encoded_power[128], encoded_one[128];
		bigint_to_montgomery(power, modulus, words, encoded_power);
		bigint_to_montgomery(accumulator, modulus, words, encoded_one);
		for (uint i = 0u; i < 128u; ++i) {
			power[i] = encoded_power[i]; accumulator[i] = encoded_one[i];
		}
		negative_inverse = bigint_montgomery_factor(modulus[0]);
	}
	uint bits = 0u;
	for (uint i = 0u; i < words; ++i) {
		if (exponent[i] != 0u) bits = i * 32u + uint(firstbithigh(exponent[i])) + 1u;
	}
	[loop]
	for (uint bit = 0u; bit < bits; ++bit) {
		uint product[128];
		if (((exponent[bit / 32u] >> (bit % 32u)) & 1u) != 0u) {
			if (montgomery)
				bigint_montgomery_product(accumulator, power, modulus, words, negative_inverse, product);
			else bigint_multiply_mod(accumulator, power, modulus, words, product);
			for (uint i = 0u; i < 128u; ++i) accumulator[i] = product[i];
		}
		if (bit + 1u < bits) {
			if (montgomery)
				bigint_montgomery_product(power, power, modulus, words, negative_inverse, product);
			else bigint_multiply_mod(power, power, modulus, words, product);
			for (uint i = 0u; i < 128u; ++i) power[i] = product[i];
		}
	}
	if (montgomery) {
		uint ordinary_one[128];
		bigint_reduce(one, modulus, words, ordinary_one);
		bigint_montgomery_product(accumulator, ordinary_one, modulus, words, negative_inverse, result);
	} else for (uint i = 0u; i < 128u; ++i) result[i] = accumulator[i];
	return true;
}
