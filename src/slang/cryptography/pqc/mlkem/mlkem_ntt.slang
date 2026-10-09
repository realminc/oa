// ML-KEM fixed 256-coefficient NTT, FIPS 203 Algorithms 9-12, Appendix A.
// Adaptation: new standards-derived implementation; no live OA C++ ML-KEM donor.
// All coefficients and twiddles are ordinary canonical residues [0,3329).
// Unlike ML-DSA, the transform stops at length 2: 128 quadratic residues.
// Private public-data laboratory; hardware/security/performance unqualified.
import mlkem_field;

// Entry i = 17^BitRev7(i) mod q. Index zero is unused by the butterflies.
static const uint MLKEM_ZETA[128] = {
	1u, 1729u, 2580u, 3289u, 2642u, 630u, 1897u, 848u,
	1062u, 1919u, 193u, 797u, 2786u, 3260u, 569u, 1746u,
	296u, 2447u, 1339u, 1476u, 3046u, 56u, 2240u, 1333u,
	1426u, 2094u, 535u, 2882u, 2393u, 2879u, 1974u, 821u,
	289u, 331u, 3253u, 1756u, 1197u, 2304u, 2277u, 2055u,
	650u, 1977u, 2513u, 632u, 2865u, 33u, 1320u, 1915u,
	2319u, 1435u, 807u, 452u, 1438u, 2868u, 1534u, 2402u,
	2647u, 2617u, 1481u, 648u, 2474u, 3110u, 1227u, 910u,
	17u, 2761u, 583u, 2649u, 1637u, 723u, 2288u, 1100u,
	1409u, 2662u, 3281u, 233u, 756u, 2156u, 3015u, 3050u,
	1703u, 1651u, 2789u, 1789u, 1847u, 952u, 1461u, 2687u,
	939u, 2308u, 2437u, 2388u, 733u, 2337u, 268u, 641u,
	1584u, 2298u, 2037u, 3220u, 375u, 2549u, 2090u, 1645u,
	1063u, 319u, 2773u, 757u, 2099u, 561u, 2466u, 2594u,
	2804u, 1092u, 403u, 1026u, 1143u, 2150u, 2775u, 886u,
	1722u, 1212u, 1874u, 1029u, 2110u, 2935u, 885u, 2154u
};

void mlkem_ntt(inout uint poly[256]) {
	uint index = 1u;
	for (uint length = 128u; length >= 2u; length >>= 1u) {
		for (uint start = 0u; start < 256u; start += 2u * length) {
			uint zeta = MLKEM_ZETA[index++];
			for (uint j = start; j < start + length; ++j) {
				uint product = mlkem_mul(zeta, poly[j + length]);
				uint value = poly[j];
				poly[j] = mlkem_add(value, product);
				poly[j + length] = mlkem_sub(value, product);
			}
		}
	}
}

void mlkem_intt(inout uint poly[256]) {
	uint index = 127u;
	for (uint length = 2u; length <= 128u; length <<= 1u) {
		for (uint start = 0u; start < 256u; start += 2u * length) {
			uint zeta = MLKEM_ZETA[index--];
			for (uint j = start; j < start + length; ++j) {
				uint a = poly[j], b = poly[j + length];
				poly[j] = mlkem_add(a, b);
				poly[j + length] = mlkem_mul(zeta, mlkem_sub(b, a));
			}
		}
	}
	// 128^-1 mod q, not 256^-1 and not a Montgomery conversion.
	for (uint j = 0u; j < 256u; ++j) poly[j] = mlkem_mul(poly[j], 3303u);
}

uint2 mlkem_base_multiply(uint2 a, uint2 b, uint gamma) {
	uint even = mlkem_add(mlkem_mul(a.x, b.x), mlkem_mul(mlkem_mul(a.y, b.y), gamma));
	uint odd = mlkem_add(mlkem_mul(a.x, b.y), mlkem_mul(a.y, b.x));
	return uint2(even, odd);
}

// In-place left operand; right must be a distinct array. No coefficient-wise
// scalar product: every pair multiplies modulo X^2 - gamma.
void mlkem_multiply_ntts(inout uint left[256], uint right[256]) {
	for (uint pair = 0u; pair < 128u; ++pair) {
		// 17^(2*BitRev7(pair)+1); Appendix A's alternating +/- final-stage roots.
		uint gamma = MLKEM_ZETA[64u + pair / 2u];
		if ((pair & 1u) != 0u) gamma = mlkem_neg(gamma);
		uint j = 2u * pair;
		uint2 product = mlkem_base_multiply(uint2(left[j], left[j + 1u]),
			uint2(right[j], right[j + 1u]), gamma);
		left[j] = product.x;
		left[j + 1u] = product.y;
	}
}
