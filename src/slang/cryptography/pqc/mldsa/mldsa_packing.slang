// ML-DSA polynomial packing and unpacking (FIPS 204).
//
// Implements the bit-packing algorithms from FIPS 204 §5.2 as direct device
// helper routines used by the verify shader.  All buffer accesses are
// byte-addressed via a caller-provided base offset into a storage buffer.
//
// Algorithms implemented (FIPS 204 names):
//   SimpleBitPack  / SimplebitUnpack  — for general bounded-coefficient polynomials
//   BitPack / BitUnpack               — for centered-coefficient polynomials
//   HintBitPack / HintBitUnpack       — for the hint vector h
//
// Donor reference: FIPS 204 §5.2; no OA donor shader body is reused.
// Adaptation class: new implementation.

import mldsa_field;

// All callers preflight the complete encoded range before entry. Aligned word
// loads require physical storage padded through its final four-byte word.
// No coefficient extractor reads bytes beyond its own packed polynomial.
uint read_byte(RWByteAddressBuffer buf, uint offset) {
	uint word = buf.Load(offset & ~3u);
	return (word >> ((offset & 3u) * 8u)) & 0xFFu;
}

// Little-endian bit extraction for widths 1..20. Read only the bytes needed:
// in particular, the final coefficient cannot overread into the next object.
uint unpack_unsigned_coeff(
	RWByteAddressBuffer buf, uint buf_base, uint coeff_idx, uint bits
) {
	uint bit_offset = coeff_idx * bits;
	uint byte_start = bit_offset / 8u;
	uint bit_shift = bit_offset % 8u;
	uint byte_count = (bit_shift + bits + 7u) / 8u;
	uint window = 0u;
	for (uint j = 0u; j < byte_count; ++j) {
		window |= read_byte(buf, buf_base + byte_start + j) << (j * 8u);
	}
	return (window >> bit_shift) & ((1u << bits) - 1u);
}

uint read_le_word(RWByteAddressBuffer buf, uint offset) {
	uint word = 0u;
	for (uint j = 0u; j < 4u; ++j) {
		word |= read_byte(buf, offset + j) << (j * 8u);
	}
	return word;
}

// ---------------------------------------------------------------------------
// Unpack a signed coefficient packed with range `a` (FIPS 204 BitUnpack).
// Packed value p = a - coeff; unpack as coeff = a - p.
// bits = ceil(log2(2*a + 1)).
// ---------------------------------------------------------------------------
int unpack_signed_coeff(
	RWByteAddressBuffer buf,
	uint buf_base,
	uint coeff_idx,
	uint bits,
	uint half_range    // = a in FIPS 204 notation
) {
	uint p = unpack_unsigned_coeff(buf, buf_base, coeff_idx, bits);
	return (int)half_range - (int)p;
}

// ---------------------------------------------------------------------------
// Unpack 256 coefficients of a polynomial packed with bitwidth `bits` and
// unsigned range [0, 2^bits). Writes into out_poly[256].
// ---------------------------------------------------------------------------
void mldsa_unpack_poly_unsigned(
	RWByteAddressBuffer buf,
	uint buf_base,
	uint bits,
	out uint out_poly[256]
) {
	for (uint i = 0u; i < 256u; ++i) {
		out_poly[i] = unpack_unsigned_coeff(buf, buf_base, i, bits);
	}
}

// ---------------------------------------------------------------------------
// Unpack 256 signed coefficients packed with half-range a and bitwidth bits.
// Writes into out_poly[256] as values in [0, Q) (reduced mod Q).
// ---------------------------------------------------------------------------
void mldsa_unpack_poly_signed(
	RWByteAddressBuffer buf,
	uint buf_base,
	uint bits,
	uint half_range,
	out uint out_poly[256]
) {
	for (uint i = 0u; i < 256u; ++i) {
		int c = unpack_signed_coeff(buf, buf_base, i, bits, half_range);
		// Map into [0, Q): negative values add Q.
		out_poly[i] = (c < 0) ? (uint)(c + (int)MLDSA_Q) : (uint)c;
	}
}

// FIPS 204 Table 1, Algorithms 23 and 27: ML-DSA-65 k=6, l=5,
// lambda=192, c-tilde=lambda/4=48 bytes. pk=32+6*320=1952 bytes;
// signature=48+5*640+55+6=3309 bytes. Every t1 and z bit pattern
// denotes a coefficient; canonical rejection occurs in the hint encoding.
static const uint MLDSA65_K = 6u;
static const uint MLDSA65_L = 5u;
static const uint MLDSA65_T1_BITS = 10u;
static const uint MLDSA65_T1_POLY_BYTES = 320u;
static const uint MLDSA65_GAMMA1 = 524288u;
static const uint MLDSA65_GAMMA1_BITS = 20u;
static const uint MLDSA65_GAMMA2 = 261888u;
static const uint MLDSA65_OMEGA = 55u;
static const uint MLDSA65_LAMBDA_BYTES = 48u;
static const uint MLDSA65_Z_POLY_BYTES = 640u;
static const uint MLDSA65_H_BYTES = 61u;

void mldsa65_unpack_pk(
	RWByteAddressBuffer buf, uint pk_base,
	out uint rho_out[8], out uint t1_out[6][256]
) {
	for (uint w = 0u; w < 8u; ++w) {
		rho_out[w] = read_le_word(buf, pk_base + w * 4u);
	}
	for (uint poly = 0u; poly < MLDSA65_K; ++poly) {
		mldsa_unpack_poly_unsigned(buf,
			pk_base + 32u + poly * MLDSA65_T1_POLY_BYTES,
			MLDSA65_T1_BITS, t1_out[poly]);
	}
}

void mldsa65_unpack_ctilde(
	RWByteAddressBuffer buf, uint sig_base, out uint c_tilde[12]
) {
	for (uint w = 0u; w < 12u; ++w) {
		c_tilde[w] = read_le_word(buf, sig_base + w * 4u);
	}
}

void mldsa65_unpack_z(
	RWByteAddressBuffer buf, uint sig_base, out uint z_out[5][256]
) {
	for (uint poly = 0u; poly < MLDSA65_L; ++poly) {
		mldsa_unpack_poly_signed(buf,
			sig_base + MLDSA65_LAMBDA_BYTES + poly * MLDSA65_Z_POLY_BYTES,
			MLDSA65_GAMMA1_BITS, MLDSA65_GAMMA1, z_out[poly]);
	}
}

// Decode one canonical hint polynomial; idx carries the preceding row endpoint.
// Caller preflights the complete hint encoding and supplies its FIPS omega.
bool mldsa_unpack_hint_row(
	RWByteAddressBuffer buf, uint h_base, uint omega, uint row,
	inout uint idx, out uint poly[256]
) {
	for (uint i = 0u; i < 256u; ++i) poly[i] = 0u;
	uint end = read_byte(buf, h_base + omega + row);
	if (end < idx || end > omega) return false;
	uint start = idx;
	while (idx < end) {
		uint pos = read_byte(buf, h_base + idx);
		if (idx > start && pos <= read_byte(buf, h_base + idx - 1u)) return false;
		poly[pos] = 1u;
		++idx;
	}
	return true;
}

bool mldsa_hint_tail_valid(RWByteAddressBuffer buf, uint h_base, uint idx, uint omega) {
	for (uint i = idx; i < omega; ++i)
		if (read_byte(buf, h_base + i) != 0u) return false;
	return true;
}

bool mldsa65_unpack_h(RWByteAddressBuffer buf, uint sig_base, out uint h_out[6][256]) {
	uint h_base = sig_base + MLDSA65_LAMBDA_BYTES + MLDSA65_L * MLDSA65_Z_POLY_BYTES;
	// Decoder callers may inspect outputs after canonicality rejection. Keep
	// later rows initialized even when an earlier row fails before they are visited.
	for (uint row = 0u; row < MLDSA65_K; ++row)
		for (uint i = 0u; i < 256u; ++i) h_out[row][i] = 0u;
	uint idx = 0u;
	for (uint row = 0u; row < MLDSA65_K; ++row)
		if (!mldsa_unpack_hint_row(buf, h_base, MLDSA65_OMEGA, row, idx, h_out[row])) return false;
	return mldsa_hint_tail_valid(buf, h_base, idx, MLDSA65_OMEGA);
}

// ---------------------------------------------------------------------------
// Norm check helpers — FIPS 204 §4.2.
// Returns true if all coefficients satisfy the bound.
// ---------------------------------------------------------------------------
bool mldsa_check_norm(uint poly[256], uint bound) {
	for (uint i = 0u; i < 256u; ++i) {
		int c = mldsa_centered(poly[i]);
		uint abs_c = (c < 0) ? (uint)(-c) : (uint)c;
		if (abs_c >= bound) return false;
	}
	return true;
}
