// SHA-384, SHA-512 and SHA-512/224/256, FIPS 180-4 §§4–6.
// Donor audit: OA C++ FnHash schema/source/tests/docs have no SHA-2 implementation.
// Adaptation class: new standard formula; reuse OA uint2 rotation and packed-byte transport.
// Numeric lanes are low/high u32 words, not native u64 or dense message dtype.
// Private helpers: formula proofs do not establish GPU or secret-operation qualification.
import keccak;
import sha2_u32;
import storage;

static const uint2 SHA2_U32_PAIR_K[80] = {
	uint2(0xd728ae22u, 0x428a2f98u),
	uint2(0x23ef65cdu, 0x71374491u),
	uint2(0xec4d3b2fu, 0xb5c0fbcfu),
	uint2(0x8189dbbcu, 0xe9b5dba5u),
	uint2(0xf348b538u, 0x3956c25bu),
	uint2(0xb605d019u, 0x59f111f1u),
	uint2(0xaf194f9bu, 0x923f82a4u),
	uint2(0xda6d8118u, 0xab1c5ed5u),
	uint2(0xa3030242u, 0xd807aa98u),
	uint2(0x45706fbeu, 0x12835b01u),
	uint2(0x4ee4b28cu, 0x243185beu),
	uint2(0xd5ffb4e2u, 0x550c7dc3u),
	uint2(0xf27b896fu, 0x72be5d74u),
	uint2(0x3b1696b1u, 0x80deb1feu),
	uint2(0x25c71235u, 0x9bdc06a7u),
	uint2(0xcf692694u, 0xc19bf174u),
	uint2(0x9ef14ad2u, 0xe49b69c1u),
	uint2(0x384f25e3u, 0xefbe4786u),
	uint2(0x8b8cd5b5u, 0x0fc19dc6u),
	uint2(0x77ac9c65u, 0x240ca1ccu),
	uint2(0x592b0275u, 0x2de92c6fu),
	uint2(0x6ea6e483u, 0x4a7484aau),
	uint2(0xbd41fbd4u, 0x5cb0a9dcu),
	uint2(0x831153b5u, 0x76f988dau),
	uint2(0xee66dfabu, 0x983e5152u),
	uint2(0x2db43210u, 0xa831c66du),
	uint2(0x98fb213fu, 0xb00327c8u),
	uint2(0xbeef0ee4u, 0xbf597fc7u),
	uint2(0x3da88fc2u, 0xc6e00bf3u),
	uint2(0x930aa725u, 0xd5a79147u),
	uint2(0xe003826fu, 0x06ca6351u),
	uint2(0x0a0e6e70u, 0x14292967u),
	uint2(0x46d22ffcu, 0x27b70a85u),
	uint2(0x5c26c926u, 0x2e1b2138u),
	uint2(0x5ac42aedu, 0x4d2c6dfcu),
	uint2(0x9d95b3dfu, 0x53380d13u),
	uint2(0x8baf63deu, 0x650a7354u),
	uint2(0x3c77b2a8u, 0x766a0abbu),
	uint2(0x47edaee6u, 0x81c2c92eu),
	uint2(0x1482353bu, 0x92722c85u),
	uint2(0x4cf10364u, 0xa2bfe8a1u),
	uint2(0xbc423001u, 0xa81a664bu),
	uint2(0xd0f89791u, 0xc24b8b70u),
	uint2(0x0654be30u, 0xc76c51a3u),
	uint2(0xd6ef5218u, 0xd192e819u),
	uint2(0x5565a910u, 0xd6990624u),
	uint2(0x5771202au, 0xf40e3585u),
	uint2(0x32bbd1b8u, 0x106aa070u),
	uint2(0xb8d2d0c8u, 0x19a4c116u),
	uint2(0x5141ab53u, 0x1e376c08u),
	uint2(0xdf8eeb99u, 0x2748774cu),
	uint2(0xe19b48a8u, 0x34b0bcb5u),
	uint2(0xc5c95a63u, 0x391c0cb3u),
	uint2(0xe3418acbu, 0x4ed8aa4au),
	uint2(0x7763e373u, 0x5b9cca4fu),
	uint2(0xd6b2b8a3u, 0x682e6ff3u),
	uint2(0x5defb2fcu, 0x748f82eeu),
	uint2(0x43172f60u, 0x78a5636fu),
	uint2(0xa1f0ab72u, 0x84c87814u),
	uint2(0x1a6439ecu, 0x8cc70208u),
	uint2(0x23631e28u, 0x90befffau),
	uint2(0xde82bde9u, 0xa4506cebu),
	uint2(0xb2c67915u, 0xbef9a3f7u),
	uint2(0xe372532bu, 0xc67178f2u),
	uint2(0xea26619cu, 0xca273eceu),
	uint2(0x21c0c207u, 0xd186b8c7u),
	uint2(0xcde0eb1eu, 0xeada7dd6u),
	uint2(0xee6ed178u, 0xf57d4f7fu),
	uint2(0x72176fbau, 0x06f067aau),
	uint2(0xa2c898a6u, 0x0a637dc5u),
	uint2(0xbef90daeu, 0x113f9804u),
	uint2(0x131c471bu, 0x1b710b35u),
	uint2(0x23047d84u, 0x28db77f5u),
	uint2(0x40c72493u, 0x32caab7bu),
	uint2(0x15c9bebcu, 0x3c9ebe0au),
	uint2(0x9c100d4cu, 0x431d67c4u),
	uint2(0xcb3e42b6u, 0x4cc5d4beu),
	uint2(0xfc657e2au, 0x597f299cu),
	uint2(0x3ad6faecu, 0x5fcb6fabu),
	uint2(0x4a475817u, 0x6c44198cu)
};

uint2 sha2_u32_pair_add(uint2 left, uint2 right) {
	uint low = left.x + right.x;
	return uint2(low, left.y + right.y + (low < left.x ? 1u : 0u));
}

uint2 sha2_u32_pair_rotr(uint2 value, uint shift) {
	return rotl64(value, (64u - (shift & 63u)) & 63u);
}

uint2 sha2_u32_pair_shr(uint2 value, uint shift) {
	if (shift == 0u) return value;
	if (shift >= 64u) return uint2(0u, 0u);
	if (shift >= 32u) return uint2(value.y >> (shift - 32u), 0u);
	return uint2((value.x >> shift) | (value.y << (32u - shift)), value.y >> shift);
}

void sha2_u32_pair_compress(inout uint2 state[8], uint2 block[16]) {
	uint2 schedule[16];
	for (uint i = 0u; i < 16u; ++i) schedule[i] = block[i];
	uint2 a = state[0], b = state[1], c = state[2], d = state[3];
	uint2 e = state[4], f = state[5], g = state[6], h = state[7];
	for (uint round = 0u; round < 80u; ++round) {
		uint slot = round & 15u;
		if (round >= 16u) {
			uint2 x = schedule[(round - 15u) & 15u];
			uint2 y = schedule[(round - 2u) & 15u];
			uint2 sigma0 = sha2_u32_pair_rotr(x, 1u) ^ sha2_u32_pair_rotr(x, 8u) ^ sha2_u32_pair_shr(x, 7u);
			uint2 sigma1 = sha2_u32_pair_rotr(y, 19u) ^ sha2_u32_pair_rotr(y, 61u) ^ sha2_u32_pair_shr(y, 6u);
			schedule[slot] = sha2_u32_pair_add(sha2_u32_pair_add(schedule[slot], sigma0),
				sha2_u32_pair_add(schedule[(round - 7u) & 15u], sigma1));
		}
		uint2 sum1 = sha2_u32_pair_rotr(e, 14u) ^ sha2_u32_pair_rotr(e, 18u) ^ sha2_u32_pair_rotr(e, 41u);
		uint2 choose = (e & f) ^ (~e & g);
		uint2 temp1 = sha2_u32_pair_add(sha2_u32_pair_add(h, sum1), choose);
		temp1 = sha2_u32_pair_add(temp1, sha2_u32_pair_add(SHA2_U32_PAIR_K[round], schedule[slot]));
		uint2 sum0 = sha2_u32_pair_rotr(a, 28u) ^ sha2_u32_pair_rotr(a, 34u) ^ sha2_u32_pair_rotr(a, 39u);
		uint2 majority = (a & b) ^ (a & c) ^ (b & c);
		uint2 temp2 = sha2_u32_pair_add(sum0, majority);
		h = g; g = f; f = e; e = sha2_u32_pair_add(d, temp1);
		d = c; c = b; b = a; a = sha2_u32_pair_add(temp1, temp2);
	}
	state[0] = sha2_u32_pair_add(state[0], a); state[1] = sha2_u32_pair_add(state[1], b);
	state[2] = sha2_u32_pair_add(state[2], c); state[3] = sha2_u32_pair_add(state[3], d);
	state[4] = sha2_u32_pair_add(state[4], e); state[5] = sha2_u32_pair_add(state[5], f);
	state[6] = sha2_u32_pair_add(state[6], g); state[7] = sha2_u32_pair_add(state[7], h);
}

bool sha2_u32_pair_digest(
	RWByteAddressBuffer buf,
	uint msg_base,
	uint msg_len,
	uint bits,
	inout uint digest[16]
) {
	uint extent;
	buf.GetDimensions(extent);
	if ((bits != 224u && bits != 256u && bits != 384u && bits != 512u)
		|| (extent & 3u) != 0u || msg_base > extent || msg_len > extent - msg_base) return false;
	uint2 state[8];
	if (bits == 224u) {
		state[0] = uint2(0x19544da2u, 0x8c3d37c8u);
		state[1] = uint2(0x89dcd4d6u, 0x73e19966u);
		state[2] = uint2(0x32ff9c82u, 0x1dfab7aeu);
		state[3] = uint2(0x582f9fcfu, 0x679dd514u);
		state[4] = uint2(0x7bd44da8u, 0x0f6d2b69u);
		state[5] = uint2(0x04c48942u, 0x77e36f73u);
		state[6] = uint2(0x6a1d36c8u, 0x3f9d85a8u);
		state[7] = uint2(0x91d692a1u, 0x1112e6adu);
	} else if (bits == 256u) {
		state[0] = uint2(0xfc2bf72cu, 0x22312194u);
		state[1] = uint2(0xc84c64c2u, 0x9f555fa3u);
		state[2] = uint2(0x6f53b151u, 0x2393b86bu);
		state[3] = uint2(0x5940eabdu, 0x96387719u);
		state[4] = uint2(0xa88effe3u, 0x96283ee2u);
		state[5] = uint2(0x53863992u, 0xbe5e1e25u);
		state[6] = uint2(0x2c85b8aau, 0x2b0199fcu);
		state[7] = uint2(0x81c52ca2u, 0x0eb72ddcu);
	} else if (bits == 384u) {
		state[0] = uint2(0xc1059ed8u, 0xcbbb9d5du);
		state[1] = uint2(0x367cd507u, 0x629a292au);
		state[2] = uint2(0x3070dd17u, 0x9159015au);
		state[3] = uint2(0xf70e5939u, 0x152fecd8u);
		state[4] = uint2(0xffc00b31u, 0x67332667u);
		state[5] = uint2(0x68581511u, 0x8eb44a87u);
		state[6] = uint2(0x64f98fa7u, 0xdb0c2e0du);
		state[7] = uint2(0xbefa4fa4u, 0x47b5481du);
	} else if (bits == 512u) {
		state[0] = uint2(0xf3bcc908u, 0x6a09e667u);
		state[1] = uint2(0x84caa73bu, 0xbb67ae85u);
		state[2] = uint2(0xfe94f82bu, 0x3c6ef372u);
		state[3] = uint2(0x5f1d36f1u, 0xa54ff53au);
		state[4] = uint2(0xade682d1u, 0x510e527fu);
		state[5] = uint2(0x2b3e6c1fu, 0x9b05688cu);
		state[6] = uint2(0xfb41bd6bu, 0x1f83d9abu);
		state[7] = uint2(0x137e2179u, 0x5be0cd19u);
	}
	uint2 block[16];
	uint offset = 0u;
	uint remaining = msg_len;
	while (remaining >= 128u) {
		for (uint word = 0u; word < 32u; ++word) {
			uint address = msg_base + offset + word * 4u;
			uint value;
			if ((address & 3u) == 0u) value = sha2_u32_swap(buf.Load(address));
			else value = (load_u8(buf, address) << 24u) | (load_u8(buf, address + 1u) << 16u)
				| (load_u8(buf, address + 2u) << 8u) | load_u8(buf, address + 3u);
			if ((word & 1u) == 0u) block[word / 2u].y = value;
			else block[word / 2u].x = value;
		}
		sha2_u32_pair_compress(state, block);
		offset += 128u;
		remaining -= 128u;
	}
	for (uint word = 0u; word < 16u; ++word) block[word] = uint2(0u, 0u);
	for (uint byte = 0u; byte <= remaining; ++byte) {
		uint value = byte == remaining ? 0x80u : load_u8(buf, msg_base + offset + byte);
		value <<= 24u - (byte & 3u) * 8u;
		if ((byte & 7u) < 4u) block[byte / 8u].y |= value;
		else block[byte / 8u].x |= value;
	}
	if (remaining >= 112u) {
		sha2_u32_pair_compress(state, block);
		for (uint word = 0u; word < 16u; ++word) block[word] = uint2(0u, 0u);
	}
	// The high 64 bits of the 128-bit bit count are zero for this checked u32 byte range.
	block[15] = uint2(msg_len << 3u, msg_len >> 29u);
	sha2_u32_pair_compress(state, block);
	for (uint word = 0u; word < bits / 32u; ++word) {
		uint value = (word & 1u) == 0u ? state[word / 2u].y : state[word / 2u].x;
		digest[word] = sha2_u32_swap(value);
	}
	return true;
}

bool sha2_512_224(RWByteAddressBuffer buf, uint msg_base, uint msg_len, inout uint digest[7]) {
	uint full_digest[16];
	if (!sha2_u32_pair_digest(buf, msg_base, msg_len, 224u, full_digest)) return false;
	for (uint word = 0u; word < 7u; ++word) digest[word] = full_digest[word];
	return true;
}

bool sha2_512_256(RWByteAddressBuffer buf, uint msg_base, uint msg_len, inout uint digest[8]) {
	uint full_digest[16];
	if (!sha2_u32_pair_digest(buf, msg_base, msg_len, 256u, full_digest)) return false;
	for (uint word = 0u; word < 8u; ++word) digest[word] = full_digest[word];
	return true;
}

bool sha2_384(RWByteAddressBuffer buf, uint msg_base, uint msg_len, inout uint digest[12]) {
	uint full_digest[16];
	if (!sha2_u32_pair_digest(buf, msg_base, msg_len, 384u, full_digest)) return false;
	for (uint word = 0u; word < 12u; ++word) digest[word] = full_digest[word];
	return true;
}

bool sha2_512(RWByteAddressBuffer buf, uint msg_base, uint msg_len, inout uint digest[16]) {
	uint full_digest[16];
	if (!sha2_u32_pair_digest(buf, msg_base, msg_len, 512u, full_digest)) return false;
	for (uint word = 0u; word < 16u; ++word) digest[word] = full_digest[word];
	return true;
}
