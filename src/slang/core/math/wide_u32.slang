// Exact 32x32 -> two-word product without shaderInt64.
// Donor: cryptography/pqc/mldsa/mldsa_field.slang, mul32_wide/mulhi32.
// Adaptation: mechanical extraction; preserve the independently proved formula.

uint2 mul32_wide(uint a, uint b) {
	uint a_lo = a & 0xFFFFu;
	uint a_hi = a >> 16u;
	uint b_lo = b & 0xFFFFu;
	uint b_hi = b >> 16u;

	uint lo_lo = a_lo * b_lo;                  // 0..32 bits
	uint lo_hi = a_lo * b_hi;                  // 16..48 bits
	uint hi_lo = a_hi * b_lo;                  // 16..48 bits
	uint hi_hi = a_hi * b_hi;                  // 32..64 bits

	uint mid   = (lo_lo >> 16u) + (lo_hi & 0xFFFFu) + (hi_lo & 0xFFFFu);
	uint carry = mid >> 16u;

	uint result_lo = (lo_lo & 0xFFFFu) | ((mid & 0xFFFFu) << 16u);
	uint result_hi = hi_hi + (lo_hi >> 16u) + (hi_lo >> 16u) + carry;
	return uint2(result_lo, result_hi);
}

// Return high 32 bits of a*b.
uint mulhi32(uint a, uint b) {
	return mul32_wide(a, b).y;
}
