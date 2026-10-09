// FIPS 204 fixed SHAKE-128/256 prehash digests, 32/64 bytes respectively.
// Donor: oacpp/source/cpp/lib/oa/crypto/shader/compute/shake/shake128.slang
// and shake256.slang: absorb virtual domain/padding bytes, then permute.
// Adaptation: checked one-message helper; no descriptor or output-buffer ABI.
// Reuses the donor permutation and canonical packed-byte transport.
import shake;
import keccak;
import storage;

bool pqc_shake_prehash(
	RWByteAddressBuffer messages,
	uint base,
	uint length,
	uint bits,
	inout uint digest[16]
) {
	if ((bits != 128u && bits != 256u) || !pqc_buffer_range(messages, base, length)) return false;
	uint rate = bits == 128u ? 168u : 136u;
	uint words = bits == 128u ? 8u : 16u;
	uint2 state[25];
	for (uint i = 0u; i < 25u; ++i) state[i] = uint2(0u, 0u);
	uint offset = 0u;
	uint remaining = length;
	// Include one final block even when the message ends exactly at the rate.
	while (true) {
		uint amount = min(remaining, rate);
		for (uint byte = 0u; byte < rate; ++byte) {
			uint value = byte < amount ? load_u8(messages, base + offset + byte) : 0u;
			if (amount < rate && byte == amount) value |= 0x1fu;
			if (amount < rate && byte == rate - 1u) value |= 0x80u;
			uint lane = byte / 8u;
			if ((byte & 7u) < 4u) state[lane].x ^= value << ((byte & 3u) * 8u);
			else state[lane].y ^= value << ((byte & 3u) * 8u);
		}
		keccak_f1600_u32(state);
		if (amount < rate) break;
		offset += amount;
		remaining -= amount;
	}
	// Both fixed digests fit in the first squeeze block.
	for (uint word = 0u; word < words; ++word)
		digest[word] = (word & 1u) == 0u ? state[word / 2u].x : state[word / 2u].y;
	return true;
}
