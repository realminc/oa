// Streaming public-data SHAKE-256, FIPS 202. Reuses OA's proven permutation.
// Adaptation: new composition helper; historical PQC absorb code lost position.
// Caller preflights buffer ranges and provides final-word padding. No secret
// execution or constant-time claim is attached to this helper.
import keccak;

struct PqcShake256 {
	uint2 state[25];
	uint position;
};

void pqc_shake256_init(out PqcShake256 sponge) {
	for (uint i = 0u; i < 25u; ++i) sponge.state[i] = uint2(0u, 0u);
	sponge.position = 0u;
}

void pqc_shake256_absorb_byte(inout PqcShake256 sponge, uint value) {
	uint lane = sponge.position / 8u;
	uint shift = (sponge.position % 4u) * 8u;
	if ((sponge.position % 8u) < 4u) sponge.state[lane].x ^= (value & 255u) << shift;
	else sponge.state[lane].y ^= (value & 255u) << shift;
	++sponge.position;
	if (sponge.position == 136u) {
		keccak_f1600_u32(sponge.state);
		sponge.position = 0u;
	}
}

void pqc_shake256_absorb_buffer(
	inout PqcShake256 sponge, RWByteAddressBuffer buffer, uint base, uint length
) {
	for (uint i = 0u; i < length; ++i) {
		uint address = base + i;
		uint value = buffer.Load(address & ~3u) >> ((address & 3u) * 8u);
		pqc_shake256_absorb_byte(sponge, value);
	}
}

void pqc_shake256_absorb_digest(inout PqcShake256 sponge, uint2 digest[8]) {
	for (uint i = 0u; i < 64u; ++i) {
		uint word = ((i % 8u) < 4u) ? digest[i / 8u].x : digest[i / 8u].y;
		pqc_shake256_absorb_byte(sponge, word >> ((i % 4u) * 8u));
	}
}

// Switch from absorption to squeezing once; do not absorb after this boundary.
void pqc_shake256_finalize(inout PqcShake256 sponge) {
	uint lane = sponge.position / 8u;
	uint shift = (sponge.position % 4u) * 8u;
	if ((sponge.position % 8u) < 4u) sponge.state[lane].x ^= 31u << shift;
	else sponge.state[lane].y ^= 31u << shift;
	sponge.state[16].y ^= 0x80000000u;
	keccak_f1600_u32(sponge.state);
	sponge.position = 0u;
}

// Sequential output has no fixed length or silently truncating block limit.
uint pqc_shake256_squeeze_byte(inout PqcShake256 sponge) {
	if (sponge.position == 136u) {
		keccak_f1600_u32(sponge.state);
		sponge.position = 0u;
	}
	uint position = sponge.position++;
	uint word = position % 8u < 4u ? sponge.state[position / 8u].x
		: sponge.state[position / 8u].y;
	return (word >> ((position % 4u) * 8u)) & 255u;
}

// Existing fixed-digest interface, destructive finalization exactly once.
void pqc_shake256_finish(inout PqcShake256 sponge, out uint2 digest[8]) {
	pqc_shake256_finalize(sponge);
	for (uint i = 0u; i < 8u; ++i) digest[i] = sponge.state[i];
	sponge.position = 64u;
}

// Subtraction-based checks avoid overflow before any byte-addressed access.
// Physical buffers must include their complete final word, even for odd tails.
bool pqc_buffer_range(RWByteAddressBuffer buffer, uint base, uint length) {
	uint size;
	buffer.GetDimensions(size);
	if ((size & 3u) != 0u) return false;
	return base <= size && length <= size - base;
}
