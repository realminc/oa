// CPU Keccak-f[1600], SHAKE-128/256, KMAC-256 implementation.
//
// Ported from NIST reference (public domain). Matches GPU shader bit-exact.
// State: 25 x uint64_t, little-endian lane ordering.

#include <Oa/Crypto/Keccak.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

namespace {

class SecureEraseGuard {
public:
	SecureEraseGuard(void* InData, OaUsize InSize)
		: Data_(static_cast<volatile OaByte*>(InData)), Size_(InSize) {}

	~SecureEraseGuard() {
		for (OaUsize i = 0; i < Size_; ++i) Data_[i] = 0;
	}

	SecureEraseGuard(const SecureEraseGuard&) = delete;
	SecureEraseGuard& operator=(const SecureEraseGuard&) = delete;

private:
	volatile OaByte* Data_;
	OaUsize Size_;
};

} // namespace

// Round constants for Keccak-f[1600] (24 rounds).
static const OaU64 RC[24] = {
	0x0000000000000001ULL, 0x0000000000008082ULL,
	0x800000000000808AULL, 0x8000000080008000ULL,
	0x000000000000808BULL, 0x0000000080000001ULL,
	0x8000000080008081ULL, 0x8000000000008009ULL,
	0x000000000000008AULL, 0x0000000000000088ULL,
	0x0000000080008009ULL, 0x000000008000000AULL,
	0x000000008000808BULL, 0x800000000000008BULL,
	0x8000000000008089ULL, 0x8000000000008003ULL,
	0x8000000000008002ULL, 0x8000000000000080ULL,
	0x000000000000800AULL, 0x800000008000000AULL,
	0x8000000080008081ULL, 0x8000000000008080ULL,
	0x0000000080000001ULL, 0x8000000080008008ULL,
};

// Rho rotation offsets indexed as [5*y + x].
static const OaU32 ROT[25] = {
	 0,  1, 62, 28, 27,
	36, 44,  6, 55, 20,
	 3, 10, 43, 25, 39,
	41, 45, 15, 21,  8,
	18,  2, 61, 56, 14,
};

static inline OaU64 Rotl64(OaU64 InVal, OaU32 InN) {
	if (InN == 0) {
		return InVal;
	}
	return (InVal << InN) | (InVal >> (64 - InN));
}

void OaKeccakF1600(OaU64* InOutState) {
	OaU64* s = InOutState;

	for (OaU32 round = 0; round < 24; ++round) {
		// Theta
		OaU64 c[5];
		for (OaU32 i = 0; i < 5; ++i) {
			c[i] = s[i] ^ s[5 + i] ^ s[10 + i] ^ s[15 + i] ^ s[20 + i];
		}

		OaU64 d[5];
		for (OaU32 i = 0; i < 5; ++i) {
			d[i] = c[(i + 4) % 5] ^ Rotl64(c[(i + 1) % 5], 1);
		}

		for (OaU32 i = 0; i < 25; ++i) {
			s[i] ^= d[i % 5];
		}

		// Rho + Pi (combined)
		OaU64 t[25];
		for (OaU32 i = 0; i < 25; ++i) {
			OaU32 x = i % 5;
			OaU32 y = i / 5;
			OaU32 nx = y;
			OaU32 ny = ((2 * x) + (3 * y)) % 5;
			t[(5 * ny) + nx] = Rotl64(s[i], ROT[i]);
		}

		// Chi
		for (OaU32 y = 0; y < 5; ++y) {
			OaU32 b = y * 5;
			s[b + 0] = t[b + 0] ^ (~t[b + 1] & t[b + 2]);
			s[b + 1] = t[b + 1] ^ (~t[b + 2] & t[b + 3]);
			s[b + 2] = t[b + 2] ^ (~t[b + 3] & t[b + 4]);
			s[b + 3] = t[b + 3] ^ (~t[b + 4] & t[b + 0]);
			s[b + 4] = t[b + 4] ^ (~t[b + 0] & t[b + 1]);
		}

		// Iota
		s[0] ^= RC[round];
	}
}

// XOR InLen bytes from InData into state (interpreted as little-endian bytes).
static void XorBytesIntoState(OaU64* InOutState, const OaByte* InData, OaUsize InLen) {
	OaUsize fullLanes = InLen / 8;
	for (OaUsize i = 0; i < fullLanes; ++i) {
		OaU64 lane;
		std::memcpy(&lane, InData + (i * 8), 8);
		InOutState[i] ^= lane;
	}
	OaUsize tail = InLen % 8;
	if (tail > 0) {
		OaU64 lane = 0;
		std::memcpy(&lane, InData + (fullLanes * 8), tail);
		InOutState[fullLanes] ^= lane;
	}
}

// Extract InLen bytes from state into OutData.
static void ExtractBytesFromState(const OaU64* InState, OaByte* OutData, OaUsize InLen) {
	OaUsize fullLanes = InLen / 8;
	for (OaUsize i = 0; i < fullLanes; ++i) {
		std::memcpy(OutData + (i * 8), &InState[i], 8);
	}
	OaUsize tail = InLen % 8;
	if (tail > 0) {
		std::memcpy(OutData + (fullLanes * 8), &InState[fullLanes], tail);
	}
}

// Core sponge: absorb + squeeze with configurable rate and domain byte.
static void Sponge(
	OaU32 InRate, OaByte InDomainByte,
	const OaByte* InData, OaUsize InLen,
	OaByte* OutDigest, OaUsize InOutLen)
{
	OaU64 state[25] = {};

	// Absorb
	OaUsize offset = 0;
	while (offset + InRate <= InLen) {
		XorBytesIntoState(state, InData + offset, InRate);
		OaKeccakF1600(state);
		offset += InRate;
	}

	// Final block: remaining bytes + padding
	OaUsize remaining = InLen - offset;
	OaByte lastBlock[200] = {};
	if (remaining != 0) {
		std::memcpy(lastBlock, InData + offset, remaining);
	}
	lastBlock[remaining] = InDomainByte;
	lastBlock[InRate - 1] |= 0x80;
	XorBytesIntoState(state, lastBlock, InRate);
	OaKeccakF1600(state);

	// Squeeze
	OaUsize squeezed = 0;
	while (squeezed < InOutLen) {
		OaUsize chunk = std::min(InOutLen - squeezed, static_cast<OaUsize>(InRate));
		ExtractBytesFromState(state, OutDigest + squeezed, chunk);
		squeezed += chunk;
		if (squeezed < InOutLen) {
			OaKeccakF1600(state);
		}
	}
}

void OaShake128(
	const OaByte* InData, OaUsize InLen,
	OaByte* OutDigest, OaUsize InOutLen)
{
	Sponge(168, 0x1F, InData, InLen, OutDigest, InOutLen);
}

void OaShake256(
	const OaByte* InData, OaUsize InLen,
	OaByte* OutDigest, OaUsize InOutLen)
{
	Sponge(136, 0x1F, InData, InLen, OutDigest, InOutLen);
}

// Incremental API

void OaShake128Init(OaShakeCtx& InOutCtx) {
	std::memset(&InOutCtx, 0, sizeof(OaShakeCtx));
	InOutCtx.Rate = 168;
	InOutCtx.Squeezing = false;
}

void OaShake256Init(OaShakeCtx& InOutCtx) {
	std::memset(&InOutCtx, 0, sizeof(OaShakeCtx));
	InOutCtx.Rate = 136;
	InOutCtx.Squeezing = false;
}

void OaShakeAbsorb(OaShakeCtx& InOutCtx, const OaByte* InData, OaUsize InLen) {
	OaUsize offset = 0;
	while (offset < InLen) {
		OaUsize space = InOutCtx.Rate - InOutCtx.BufLen;
		OaUsize chunk = std::min(InLen - offset, space);
		std::memcpy(InOutCtx.Buf + InOutCtx.BufLen, InData + offset, chunk);
		InOutCtx.BufLen += static_cast<OaU32>(chunk);
		offset += chunk;

		if (InOutCtx.BufLen == InOutCtx.Rate) {
			XorBytesIntoState(InOutCtx.State, InOutCtx.Buf, InOutCtx.Rate);
			OaKeccakF1600(InOutCtx.State);
			InOutCtx.BufLen = 0;
		}
	}
}

void OaShakeSqueeze(OaShakeCtx& InOutCtx, OaByte* OutData, OaUsize InOutLen) {
	if (!InOutCtx.Squeezing) {
		// Finalize: pad remaining buffer
		InOutCtx.Buf[InOutCtx.BufLen] = 0x1F;
		std::memset(InOutCtx.Buf + InOutCtx.BufLen + 1, 0,
			InOutCtx.Rate - InOutCtx.BufLen - 1);
		InOutCtx.Buf[InOutCtx.Rate - 1] |= 0x80;
		XorBytesIntoState(InOutCtx.State, InOutCtx.Buf, InOutCtx.Rate);
		OaKeccakF1600(InOutCtx.State);
		InOutCtx.BufLen = 0;
		InOutCtx.Squeezing = true;
	}

	OaUsize offset = 0;
	// Drain any leftover from a partial previous squeeze
	if (InOutCtx.BufLen > 0) {
		OaUsize avail = InOutCtx.Rate - InOutCtx.BufLen;
		OaUsize chunk = std::min(InOutLen, avail);
		OaByte rateBuf[200];
		ExtractBytesFromState(InOutCtx.State, rateBuf, InOutCtx.Rate);
		std::memcpy(OutData, rateBuf + InOutCtx.BufLen, chunk);
		InOutCtx.BufLen += static_cast<OaU32>(chunk);
		offset += chunk;
		if (InOutCtx.BufLen == InOutCtx.Rate) {
			OaKeccakF1600(InOutCtx.State);
			InOutCtx.BufLen = 0;
		}
	}

	while (offset < InOutLen) {
		OaUsize remaining = InOutLen - offset;
		if (remaining >= InOutCtx.Rate) {
			ExtractBytesFromState(InOutCtx.State, OutData + offset, InOutCtx.Rate);
			offset += InOutCtx.Rate;
			OaKeccakF1600(InOutCtx.State);
		} else {
			ExtractBytesFromState(InOutCtx.State, OutData + offset, remaining);
			InOutCtx.BufLen = static_cast<OaU32>(remaining);
			offset += remaining;
		}
	}
}

// KMAC-256 (NIST SP 800-185)
//
// KMAC256(K, X, L, S) = cSHAKE256(bytepad(encode_string(K), 136) ||
// X || right_encode(L), L, "KMAC", S). cSHAKE uses domain 0x04 and a
// bytepad(encode_string(N) || encode_string(S), rate) prefix.

// left_encode: encode integer as big-endian bytes prefixed by byte count.
static OaUsize LeftEncode(OaU64 InVal, OaByte* OutBuf) {
	OaByte tmp[9];
	OaU32 n = 0;
	if (InVal == 0) {
		tmp[0] = 0;
		n = 1;
	} else {
		OaU64 v = InVal;
		while (v > 0) {
			tmp[n++] = static_cast<OaByte>(v & 0xFF);
			v >>= 8;
		}
		// Reverse tmp[0..n-1] so it's big-endian
		for (OaU32 i = 0; i < n / 2; ++i) {
			OaByte t = tmp[i]; tmp[i] = tmp[n - 1 - i]; tmp[n - 1 - i] = t;
		}
	}
	OutBuf[0] = static_cast<OaByte>(n);
	std::memcpy(OutBuf + 1, tmp, n);
	return 1 + n;
}

// right_encode: encode integer as big-endian bytes suffixed by byte count.
static OaUsize RightEncode(OaU64 InVal, OaByte* OutBuf) {
	OaByte tmp[9];
	OaU32 n = 0;
	if (InVal == 0) {
		tmp[0] = 0;
		n = 1;
	} else {
		OaU64 v = InVal;
		while (v > 0) {
			tmp[n++] = static_cast<OaByte>(v & 0xFF);
			v >>= 8;
		}
		for (OaU32 i = 0; i < n / 2; ++i) {
			OaByte t = tmp[i]; tmp[i] = tmp[n - 1 - i]; tmp[n - 1 - i] = t;
		}
	}
	std::memcpy(OutBuf, tmp, n);
	OutBuf[n] = static_cast<OaByte>(n);
	return n + 1;
}

// encode_string(S) = left_encode(len(S)*8) || S
static OaUsize EncodeString(const OaByte* InStr, OaUsize InLen, OaByte* OutBuf) {
	OaUsize hLen = LeftEncode(InLen * 8, OutBuf);
	if (InLen != 0) {
		std::memcpy(OutBuf + hLen, InStr, InLen);
	}
	return hLen + InLen;
}

// bytepad(X, w): left_encode(w) || X || 0*pad
// We absorb the prefix directly into the sponge context.

OaStatus OaKmac256(
	const OaByte* InKey, OaUsize InKeyLen,
	const OaByte* InData, OaUsize InDataLen,
	const OaByte* InCustom, OaUsize InCustomLen,
	OaByte* OutMac, OaUsize InOutLen)
{
	const OaU32 rate = 136;
	constexpr OaUsize kMax = std::numeric_limits<OaUsize>::max();
	if ((InKey == nullptr && InKeyLen != 0) || (InData == nullptr && InDataLen != 0) ||
		(InCustom == nullptr && InCustomLen != 0) || (OutMac == nullptr && InOutLen != 0)) {
		return OaStatus::InvalidArgument("KMAC pointer is null with a non-zero length");
	}
	if (InKeyLen > std::numeric_limits<OaU64>::max() / 8 ||
		InCustomLen > std::numeric_limits<OaU64>::max() / 8 ||
		InOutLen > std::numeric_limits<OaU64>::max() / 8) {
		return OaStatus::Error(OaStatusCode::OutOfRange,
			"KMAC bit length exceeds SP 800-185 encoding");
	}
	auto PaddedSize = [](OaUsize InSize) -> OaResult<OaUsize> {
		if (InSize > std::numeric_limits<OaUsize>::max() - (rate - 1)) {
			return OaStatus::Error(OaStatusCode::OutOfRange, "KMAC encoded input is too large");
		}
		return ((InSize + rate - 1) / rate) * rate;
	};
	try {
		OaU64 state[25] = {};
		SecureEraseGuard stateGuard(state, sizeof(state));

		// cSHAKE prefix with function name N="KMAC" and customization S.
		if (InCustomLen > kMax - 32) {
				return OaStatus::Error(OaStatusCode::OutOfRange,
					"KMAC customization string is too large");
		}
		OaByte header[16];
		OaUsize headerLen = LeftEncode(rate, header);
		const OaByte kmacLabel[] = {'K', 'M', 'A', 'C'};
		headerLen += EncodeString(kmacLabel, 4, header + headerLen);
		OaByte customLenEncoding[9];
		const OaUsize customHeaderLen = LeftEncode(InCustomLen * 8, customLenEncoding);
		const OaUsize prefixRawLen = headerLen + customHeaderLen + InCustomLen;
		auto prefixSize = PaddedSize(prefixRawLen);
			if (!prefixSize) return prefixSize.GetStatus();
		std::vector<OaByte> prefix(prefixSize.GetValue(), 0);
		OaUsize pLen = 0;

		// left_encode(rate) for bytepad header
		pLen += LeftEncode(rate, prefix.data() + pLen);

		// encode_string(N) where N = "KMAC"
		pLen += EncodeString(kmacLabel, 4, prefix.data() + pLen);

		// encode_string(S) where S = InCustom
		pLen += EncodeString(InCustom, InCustomLen, prefix.data() + pLen);
		pLen = prefix.size();

		// Absorb cSHAKE prefix
		OaUsize offset = 0;
		while (offset + rate <= pLen) {
			XorBytesIntoState(state, prefix.data() + offset, rate);
			OaKeccakF1600(state);
			offset += rate;
		}
		if (offset < pLen) {
			XorBytesIntoState(state, prefix.data() + offset, pLen - offset);
		}

		// bytepad(encode_string(K), rate) for KMAC key
		if (InKeyLen > kMax - 20) {
			return OaStatus::Error(OaStatusCode::OutOfRange, "KMAC key is too large");
		}
		OaByte keyLenEncoding[9];
		const OaUsize keyHeaderLen = LeftEncode(InKeyLen * 8, keyLenEncoding);
		auto keySize = PaddedSize(LeftEncode(rate, header) + keyHeaderLen + InKeyLen);
			if (!keySize) return keySize.GetStatus();
		std::vector<OaByte> keyBuf(keySize.GetValue(), 0);
		SecureEraseGuard keyGuard(keyBuf.data(), keyBuf.size());
		OaUsize kLen = 0;
		kLen += LeftEncode(rate, keyBuf.data() + kLen);
		kLen += EncodeString(InKey, InKeyLen, keyBuf.data() + kLen);
		kLen = keyBuf.size();

		// Absorb key block
		offset = 0;
		while (offset + rate <= kLen) {
			XorBytesIntoState(state, keyBuf.data() + offset, rate);
			OaKeccakF1600(state);
			offset += rate;
		}
		if (offset < kLen) {
			XorBytesIntoState(state, keyBuf.data() + offset, kLen - offset);
		}

		// Absorb message
		offset = 0;
		while (offset + rate <= InDataLen) {
			XorBytesIntoState(state, InData + offset, rate);
			OaKeccakF1600(state);
			offset += rate;
		}

		// Final block: remaining message + right_encode(OutLen*8) + domain byte
		OaByte tail[512];
		SecureEraseGuard tailGuard(tail, sizeof(tail));
		OaUsize tLen = InDataLen - offset;
		if (tLen != 0) {
			std::memcpy(tail, InData + offset, tLen);
		}
		tLen += RightEncode(InOutLen * 8, tail + tLen);

		// cSHAKE domain: 0x04 (not SHAKE's 0x1F)
		tail[tLen] = 0x04;
		std::memset(tail + tLen + 1, 0, rate - tLen - 1);
		tail[rate - 1] |= 0x80;
		XorBytesIntoState(state, tail, rate);
		OaKeccakF1600(state);

		// Squeeze
		OaUsize squeezed = 0;
		while (squeezed < InOutLen) {
			OaUsize chunk = std::min(InOutLen - squeezed, static_cast<OaUsize>(rate));
			ExtractBytesFromState(state, OutMac + squeezed, chunk);
			squeezed += chunk;
			if (squeezed < InOutLen) {
				OaKeccakF1600(state);
			}
		}
		return OaStatus::Ok();
	} catch (const std::bad_alloc&) {
		return OaStatus::Error(OaStatusCode::ResourceExhausted,
			"KMAC temporary encoding allocation failed");
	}
}
