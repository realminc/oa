// ML-DSA fixed 256-coefficient NTT, FIPS 204 Algorithms 41 and 42.
// All polynomial coefficients, including NTT-domain coefficients, are ORDINARY
// canonical residues [0,Q). Only fixed twiddles/scaling carry Montgomery R.
// No conversion is performed at transform boundaries.
//
// Adaptation class: new standards-derived implementation. OA C++ public crypto
// delegates ML-DSA to liboqs and supplies no qualified NTT donor. The historical
// Rust draft had incorrect/short twiddle tables and mixed representations.
// Authority: FIPS 204 Section 7.5, Appendix B; zeta=1753, q=8380417.
// Component proofs: test/py/cryptography/test_pqc_components.py evaluates
// polynomials directly and compares ring products to schoolbook multiplication.

import mldsa_field;

// Entry k = 1753^BitRev8(k) * 2^32 mod Q. k=0 is unused by both transforms.
static const uint ZETA_MONT[256] = {
	4193792u, 25847u, 5771523u, 7861508u, 237124u, 7602457u, 7504169u, 466468u,
	1826347u, 2353451u, 8021166u, 6288512u, 3119733u, 5495562u, 3111497u, 2680103u,
	2725464u, 1024112u, 7300517u, 3585928u, 7830929u, 7260833u, 2619752u, 6271868u,
	6262231u, 4520680u, 6980856u, 5102745u, 1757237u, 8360995u, 4010497u, 280005u,
	2706023u, 95776u, 3077325u, 3530437u, 6718724u, 4788269u, 5842901u, 3915439u,
	4519302u, 5336701u, 3574422u, 5512770u, 3539968u, 8079950u, 2348700u, 7841118u,
	6681150u, 6736599u, 3505694u, 4558682u, 3507263u, 6239768u, 6779997u, 3699596u,
	811944u, 531354u, 954230u, 3881043u, 3900724u, 5823537u, 2071892u, 5582638u,
	4450022u, 6851714u, 4702672u, 5339162u, 6927966u, 3475950u, 2176455u, 6795196u,
	7122806u, 1939314u, 4296819u, 7380215u, 5190273u, 5223087u, 4747489u, 126922u,
	3412210u, 7396998u, 2147896u, 2715295u, 5412772u, 4686924u, 7969390u, 5903370u,
	7709315u, 7151892u, 8357436u, 7072248u, 7998430u, 1349076u, 1852771u, 6949987u,
	5037034u, 264944u, 508951u, 3097992u, 44288u, 7280319u, 904516u, 3958618u,
	4656075u, 8371839u, 1653064u, 5130689u, 2389356u, 8169440u, 759969u, 7063561u,
	189548u, 4827145u, 3159746u, 6529015u, 5971092u, 8202977u, 1315589u, 1341330u,
	1285669u, 6795489u, 7567685u, 6940675u, 5361315u, 4499357u, 4751448u, 3839961u,
	2091667u, 3407706u, 2316500u, 3817976u, 5037939u, 2244091u, 5933984u, 4817955u,
	266997u, 2434439u, 7144689u, 3513181u, 4860065u, 4621053u, 7183191u, 5187039u,
	900702u, 1859098u, 909542u, 819034u, 495491u, 6767243u, 8337157u, 7857917u,
	7725090u, 5257975u, 2031748u, 3207046u, 4823422u, 7855319u, 7611795u, 4784579u,
	342297u, 286988u, 5942594u, 4108315u, 3437287u, 5038140u, 1735879u, 203044u,
	2842341u, 2691481u, 5790267u, 1265009u, 4055324u, 1247620u, 2486353u, 1595974u,
	4613401u, 1250494u, 2635921u, 4832145u, 5386378u, 1869119u, 1903435u, 7329447u,
	7047359u, 1237275u, 5062207u, 6950192u, 7929317u, 1312455u, 3306115u, 6417775u,
	7100756u, 1917081u, 5834105u, 7005614u, 1500165u, 777191u, 2235880u, 3406031u,
	7838005u, 5548557u, 6709241u, 6533464u, 5796124u, 4656147u, 594136u, 4603424u,
	6366809u, 2432395u, 2454455u, 8215696u, 1957272u, 3369112u, 185531u, 7173032u,
	5196991u, 162844u, 1616392u, 3014001u, 810149u, 1652634u, 4686184u, 6581310u,
	5341501u, 3523897u, 3866901u, 269760u, 2213111u, 7404533u, 1717735u, 472078u,
	7953734u, 1723600u, 6577327u, 1910376u, 6712985u, 7276084u, 8119771u, 4546524u,
	5441381u, 6144432u, 7959518u, 6094090u, 183443u, 7403526u, 1612842u, 4834730u,
	7826001u, 3919660u, 8332111u, 7018208u, 3937738u, 1400424u, 7534263u, 1976782u
};

// 256^-1 * R mod Q, not the reference implementation's R^2/256 scale.
static const uint MLDSA_INV_N_MONT = 16382u;

// In-place transform, ordinary polynomial -> ordinary NTT evaluations.
void mldsa_ntt(inout uint poly[256]) {
	uint k = 0u;
	for (uint len = 128u; len >= 1u; len >>= 1u) {
		for (uint start = 0u; start < 256u; start += 2u * len) {
			uint zeta = ZETA_MONT[++k];
			for (uint j = start; j < start + len; ++j) {
				uint t = mldsa_mont_mul(zeta, poly[j + len]);
				uint a = poly[j];
				poly[j + len] = mldsa_sub(a, t);
				poly[j] = mldsa_add(a, t);
			}
		}
	}
}

// In-place inverse, ordinary NTT evaluations -> ordinary polynomial.
void mldsa_intt(inout uint poly[256]) {
	uint k = 256u;
	for (uint len = 1u; len < 256u; len <<= 1u) {
		for (uint start = 0u; start < 256u; start += 2u * len) {
			uint zeta = MLDSA_Q - ZETA_MONT[--k];
			for (uint j = start; j < start + len; ++j) {
				uint a = poly[j], b = poly[j + len];
				poly[j] = mldsa_add(a, b);
				poly[j + len] = mldsa_mont_mul(zeta, mldsa_sub(a, b));
			}
		}
	}
	for (uint i = 0u; i < 256u; ++i) {
		poly[i] = mldsa_mont_mul(MLDSA_INV_N_MONT, poly[i]);
	}
}

// Ordinary NTT-domain inputs and output; no hidden factor of R^-1.
void mldsa_poly_pointwise_ntt(uint a[256], uint b[256], out uint c[256]) {
	for (uint i = 0u; i < 256u; ++i) c[i] = mldsa_mul(a[i], b[i]);
}

void mldsa_poly_add(uint a[256], uint b[256], out uint c[256]) {
	for (uint i = 0u; i < 256u; ++i) c[i] = mldsa_add(a[i], b[i]);
}

void mldsa_poly_sub(uint a[256], uint b[256], out uint c[256]) {
	for (uint i = 0u; i < 256u; ++i) c[i] = mldsa_sub(a[i], b[i]);
}

// Scalar and polynomial are ordinary residues, regardless of transform domain.
void mldsa_poly_scalar(inout uint poly[256], uint scalar) {
	uint scaled = mldsa_to_mont(scalar);
	for (uint i = 0u; i < 256u; ++i) poly[i] = mldsa_mont_mul(scaled, poly[i]);
}
