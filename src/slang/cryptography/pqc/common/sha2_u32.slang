// Byte-oriented SHA-224/SHA-256, FIPS 180-4 §§4.1.2, 4.2.2, 5 and 6.2–6.3.
// Donor audit: OA C++ FnHash schema/source/tests/docs have no SHA-2 implementation.
// Adaptation class: new standard formula, canonical OA packed-byte transport.
// The u32 suffix identifies unsigned compression words, not message storage dtype.
// Private composable helpers, not admitted operations or GPU qualification.
// Digest words encode the standard byte sequence in little-endian packed storage.
import storage;

static const uint SHA2_U32_K[64] = {
	0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
	0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
	0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
	0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
	0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
	0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
	0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
	0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
	0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
	0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
	0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
	0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
	0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
	0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
	0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
	0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
};

uint sha2_u32_rotr(uint value, uint shift) {
	return (value >> shift) | (value << (32u - shift));
}

uint sha2_u32_swap(uint value) {
	return ((value & 0xffu) << 24u) | ((value & 0xff00u) << 8u)
		| ((value >> 8u) & 0xff00u) | (value >> 24u);
}

// High/low big-endian numeric words of the 64-bit bit count, without shaderInt64.
uint2 sha2_u32_bit_length(uint byte_length) {
	return uint2(byte_length >> 29u, byte_length << 3u);
}

void sha2_u32_compress(inout uint state[8], uint block[16]) {
	// A 16-word schedule ring replaces the standard's materialized 64-word schedule.
	uint schedule[16];
	for (uint i = 0u; i < 16u; ++i) schedule[i] = block[i];
	uint a = state[0], b = state[1], c = state[2], d = state[3];
	uint e = state[4], f = state[5], g = state[6], h = state[7];
	for (uint round = 0u; round < 64u; ++round) {
		uint slot = round & 15u;
		if (round >= 16u) {
			uint x = schedule[(round - 15u) & 15u];
			uint y = schedule[(round - 2u) & 15u];
			uint sigma0 = sha2_u32_rotr(x, 7u) ^ sha2_u32_rotr(x, 18u) ^ (x >> 3u);
			uint sigma1 = sha2_u32_rotr(y, 17u) ^ sha2_u32_rotr(y, 19u) ^ (y >> 10u);
			schedule[slot] += sigma0 + schedule[(round - 7u) & 15u] + sigma1;
		}
		uint sum1 = sha2_u32_rotr(e, 6u) ^ sha2_u32_rotr(e, 11u) ^ sha2_u32_rotr(e, 25u);
		uint choose = (e & f) ^ (~e & g);
		uint temp1 = h + sum1 + choose + SHA2_U32_K[round] + schedule[slot];
		uint sum0 = sha2_u32_rotr(a, 2u) ^ sha2_u32_rotr(a, 13u) ^ sha2_u32_rotr(a, 22u);
		uint majority = (a & b) ^ (a & c) ^ (b & c);
		uint temp2 = sum0 + majority;
		h = g; g = f; f = e; e = d + temp1;
		d = c; c = b; b = a; a = temp1 + temp2;
	}
	state[0] += a; state[1] += b; state[2] += c; state[3] += d;
	state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

bool sha2_u32_digest(
	RWByteAddressBuffer buf,
	uint msg_base,
	uint msg_len,
	uint bits,
	inout uint digest[8]
) {
	uint extent;
	buf.GetDimensions(extent);
	if ((bits != 224u && bits != 256u) || (extent & 3u) != 0u
		|| msg_base > extent || msg_len > extent - msg_base) return false;
	uint state[8];
	if (bits == 224u) {
		state[0] = 0xc1059ed8u; state[1] = 0x367cd507u;
		state[2] = 0x3070dd17u; state[3] = 0xf70e5939u;
		state[4] = 0xffc00b31u; state[5] = 0x68581511u;
		state[6] = 0x64f98fa7u; state[7] = 0xbefa4fa4u;
	} else {
		state[0] = 0x6a09e667u; state[1] = 0xbb67ae85u;
		state[2] = 0x3c6ef372u; state[3] = 0xa54ff53au;
		state[4] = 0x510e527fu; state[5] = 0x9b05688cu;
		state[6] = 0x1f83d9abu; state[7] = 0x5be0cd19u;
	}
	uint offset = 0u;
	uint remaining = msg_len;
	uint block[16];
	while (remaining >= 64u) {
		for (uint word = 0u; word < 16u; ++word) {
			uint address = msg_base + offset + word * 4u;
			if ((address & 3u) == 0u) block[word] = sha2_u32_swap(buf.Load(address));
			else block[word] = (load_u8(buf, address) << 24u)
				| (load_u8(buf, address + 1u) << 16u)
				| (load_u8(buf, address + 2u) << 8u) | load_u8(buf, address + 3u);
		}
		sha2_u32_compress(state, block);
		offset += 64u;
		remaining -= 64u;
	}
	for (uint word = 0u; word < 16u; ++word) block[word] = 0u;
	for (uint byte = 0u; byte < remaining; ++byte) {
		block[byte / 4u] |= load_u8(buf, msg_base + offset + byte) << (24u - (byte & 3u) * 8u);
	}
	block[remaining / 4u] |= 0x80u << (24u - (remaining & 3u) * 8u);
	if (remaining >= 56u) {
		sha2_u32_compress(state, block);
		for (uint word = 0u; word < 16u; ++word) block[word] = 0u;
	}
	uint2 bit_length = sha2_u32_bit_length(msg_len);
	block[14] = bit_length.x;
	block[15] = bit_length.y;
	sha2_u32_compress(state, block);
	for (uint word = 0u; word < bits / 32u; ++word) digest[word] = sha2_u32_swap(state[word]);
	return true;
}

bool sha2_256(RWByteAddressBuffer buf, uint msg_base, uint msg_len, inout uint digest[8]) {
	return sha2_u32_digest(buf, msg_base, msg_len, 256u, digest);
}

bool sha2_224(RWByteAddressBuffer buf, uint msg_base, uint msg_len, inout uint digest[7]) {
	uint full_digest[8];
	if (!sha2_u32_digest(buf, msg_base, msg_len, 224u, full_digest)) return false;
	for (uint word = 0u; word < 7u; ++word) digest[word] = full_digest[word];
	return true;
}
