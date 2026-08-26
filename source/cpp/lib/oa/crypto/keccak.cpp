// CPU Keccak-f[1600], SHAKE-128/256, KMAC-256 implementation.
//
// Ported from NIST reference (public domain). matches GPU shader bit-exact.
// State: 25 x uint64_t, little-endian lane ordering.

#include <oa/crypto/keccak.h>
#include <oa/core/memory.h>
#include <oa/core/std/algo.h>
#include <oa/core/std/allocator.h>
#include <oa/core/std/limits.h>
#include <oa/core/std/utility.h>

namespace {

class SecureEraseGuard {
public:
	SecureEraseGuard(void* inData, oa::Usize inSize)
		: data_(static_cast<volatile oa::Byte*>(inData)), size_(inSize) {}

	~SecureEraseGuard() {
		for (oa::Usize i = 0; i < size_; ++i) data_[i] = 0;
	}

	SecureEraseGuard(const SecureEraseGuard&) = delete;
	SecureEraseGuard& operator=(const SecureEraseGuard&) = delete;

private:
	volatile oa::Byte* data_;
	oa::Usize size_;
};

class SecureByteBuffer {
public:
	SecureByteBuffer() = default;
	SecureByteBuffer(const SecureByteBuffer&) = delete;
	SecureByteBuffer& operator=(const SecureByteBuffer&) = delete;

	SecureByteBuffer(SecureByteBuffer&& inOther) noexcept
		: data_(inOther.data_), size_(inOther.size_) {
		inOther.data_ = nullptr;
		inOther.size_ = 0;
	}

	SecureByteBuffer& operator=(SecureByteBuffer&& inOther) noexcept {
		if (this != &inOther) {
			reset();
			data_ = inOther.data_;
			size_ = inOther.size_;
			inOther.data_ = nullptr;
			inOther.size_ = 0;
		}
		return *this;
	}

	~SecureByteBuffer() { reset(); }

	[[nodiscard]] static oa::Result<SecureByteBuffer> create(oa::Usize inSize) {
		const oa::AllocationResult allocation = oa::tryAllocArray(
			inSize, sizeof(oa::Byte), alignof(oa::Byte));
		if (allocation.isError()) {
			return oa::Status::error(
				oa::StatusCode::ResourceExhausted,
				"KMAC temporary encoding allocation failed");
		}
		SecureByteBuffer out;
		out.data_ = static_cast<oa::Byte*>(allocation.data);
		out.size_ = inSize;
		oa::memzero(out.data_, out.size_);
		return oa::move(out);
	}

	[[nodiscard]] oa::Byte* data() noexcept { return data_; }
	[[nodiscard]] const oa::Byte* data() const noexcept { return data_; }
	[[nodiscard]] oa::Usize size() const noexcept { return size_; }

private:
	void reset() noexcept {
		if (data_ == nullptr) return;
		volatile oa::Byte* secure = data_;
		for (oa::Usize i = 0; i < size_; ++i) secure[i] = 0;
		oa::freeBytes(data_, alignof(oa::Byte));
		data_ = nullptr;
		size_ = 0;
	}

	oa::Byte* data_ = nullptr;
	oa::Usize size_ = 0;
};

} // namespace

namespace oa {

// Round constants for Keccak-f[1600] (24 rounds).
static const oa::U64 RC[24] = {
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
static const oa::U32 ROT[25] = {
	 0,  1, 62, 28, 27,
	36, 44,  6, 55, 20,
	 3, 10, 43, 25, 39,
	41, 45, 15, 21,  8,
	18,  2, 61, 56, 14,
};

static inline oa::U64 rotl64(oa::U64 inVal, oa::U32 inN) {
	if (inN == 0) {
		return inVal;
	}
	return (inVal << inN) | (inVal >> (64 - inN));
}

void keccakF1600(oa::U64* inOutState) {
	oa::U64* s = inOutState;

	for (oa::U32 round = 0; round < 24; ++round) {
		// theta
		oa::U64 c[5];
		for (oa::U32 i = 0; i < 5; ++i) {
			c[i] = s[i] ^ s[5 + i] ^ s[10 + i] ^ s[15 + i] ^ s[20 + i];
		}

		oa::U64 d[5];
		for (oa::U32 i = 0; i < 5; ++i) {
			d[i] = c[(i + 4) % 5] ^ rotl64(c[(i + 1) % 5], 1);
		}

		for (oa::U32 i = 0; i < 25; ++i) {
			s[i] ^= d[i % 5];
		}

		// Rho + pi (combined)
		oa::U64 t[25];
		for (oa::U32 i = 0; i < 25; ++i) {
			oa::U32 x = i % 5;
			oa::U32 y = i / 5;
			oa::U32 nx = y;
			oa::U32 ny = ((2 * x) + (3 * y)) % 5;
			t[(5 * ny) + nx] = rotl64(s[i], ROT[i]);
		}

		// Chi
		for (oa::U32 y = 0; y < 5; ++y) {
			oa::U32 b = y * 5;
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

// XOR inLen bytes from inData into state (interpreted as little-endian bytes).
static void xorBytesIntoState(oa::U64* inOutState, const oa::Byte* inData, oa::Usize inLen) {
	oa::Usize fullLanes = inLen / 8;
	for (oa::Usize i = 0; i < fullLanes; ++i) {
		oa::U64 lane;
		oa::memcpy(&lane, inData + (i * 8), 8);
		inOutState[i] ^= lane;
	}
	oa::Usize tail = inLen % 8;
	if (tail > 0) {
		oa::U64 lane = 0;
		oa::memcpy(&lane, inData + (fullLanes * 8), tail);
		inOutState[fullLanes] ^= lane;
	}
}

// Extract inLen bytes from state into outData.
static void extractBytesFromState(const oa::U64* inState, oa::Byte* outData, oa::Usize inLen) {
	oa::Usize fullLanes = inLen / 8;
	for (oa::Usize i = 0; i < fullLanes; ++i) {
		oa::memcpy(outData + (i * 8), &inState[i], 8);
	}
	oa::Usize tail = inLen % 8;
	if (tail > 0) {
		oa::memcpy(outData + (fullLanes * 8), &inState[fullLanes], tail);
	}
}

// Core sponge: absorb + squeeze with configurable rate and domain byte.
static void sponge(
	oa::U32 inRate, oa::Byte inDomainByte,
	const oa::Byte* inData, oa::Usize inLen,
	oa::Byte* outDigest, oa::Usize inOutLen)
{
	oa::U64 state[25] = {};

	// Absorb
	oa::Usize offset = 0;
	while (offset + inRate <= inLen) {
		xorBytesIntoState(state, inData + offset, inRate);
		oa::keccakF1600(state);
		offset += inRate;
	}

	// Final block: remaining bytes + padding
	oa::Usize remaining = inLen - offset;
	oa::Byte lastBlock[200] = {};
	if (remaining != 0) {
		oa::memcpy(lastBlock, inData + offset, remaining);
	}
	lastBlock[remaining] = inDomainByte;
	lastBlock[inRate - 1] |= 0x80;
	xorBytesIntoState(state, lastBlock, inRate);
	oa::keccakF1600(state);

	// squeeze
	oa::Usize squeezed = 0;
	while (squeezed < inOutLen) {
		oa::Usize chunk = oa::min(inOutLen - squeezed, static_cast<oa::Usize>(inRate));
		extractBytesFromState(state, outDigest + squeezed, chunk);
		squeezed += chunk;
		if (squeezed < inOutLen) {
			oa::keccakF1600(state);
		}
	}
}

void shake128(
	const oa::Byte* inData, oa::Usize inLen,
	oa::Byte* outDigest, oa::Usize inOutLen)
{
	sponge(168, 0x1F, inData, inLen, outDigest, inOutLen);
}

void shake256(
	const oa::Byte* inData, oa::Usize inLen,
	oa::Byte* outDigest, oa::Usize inOutLen)
{
	sponge(136, 0x1F, inData, inLen, outDigest, inOutLen);
}

// Incremental API

void shake128Init(ShakeContext& inOutCtx) {
	oa::memzero(&inOutCtx, sizeof(ShakeContext));
	inOutCtx.rate = 168;
	inOutCtx.squeezing = false;
}

void shake256Init(ShakeContext& inOutCtx) {
	oa::memzero(&inOutCtx, sizeof(ShakeContext));
	inOutCtx.rate = 136;
	inOutCtx.squeezing = false;
}

void shakeAbsorb(ShakeContext& inOutCtx, const oa::Byte* inData, oa::Usize inLen) {
	oa::Usize offset = 0;
	while (offset < inLen) {
		oa::Usize space = inOutCtx.rate - inOutCtx.bufLen;
		oa::Usize chunk = oa::min(inLen - offset, space);
		oa::memcpy(inOutCtx.buf + inOutCtx.bufLen, inData + offset, chunk);
		inOutCtx.bufLen += static_cast<oa::U32>(chunk);
		offset += chunk;

		if (inOutCtx.bufLen == inOutCtx.rate) {
			xorBytesIntoState(inOutCtx.state, inOutCtx.buf, inOutCtx.rate);
			oa::keccakF1600(inOutCtx.state);
			inOutCtx.bufLen = 0;
		}
	}
}

void shakeSqueeze(ShakeContext& inOutCtx, oa::Byte* outData, oa::Usize inOutLen) {
	if (!inOutCtx.squeezing) {
		// finalize: pad remaining buffer
		inOutCtx.buf[inOutCtx.bufLen] = 0x1F;
		oa::memzero(inOutCtx.buf + inOutCtx.bufLen + 1,
			inOutCtx.rate - inOutCtx.bufLen - 1);
		inOutCtx.buf[inOutCtx.rate - 1] |= 0x80;
		xorBytesIntoState(inOutCtx.state, inOutCtx.buf, inOutCtx.rate);
		oa::keccakF1600(inOutCtx.state);
		inOutCtx.bufLen = 0;
		inOutCtx.squeezing = true;
	}

	oa::Usize offset = 0;
	// drain any leftover from a partial previous squeeze
	if (inOutCtx.bufLen > 0) {
		oa::Usize avail = inOutCtx.rate - inOutCtx.bufLen;
		oa::Usize chunk = oa::min(inOutLen, avail);
		oa::Byte rateBuf[200];
		extractBytesFromState(inOutCtx.state, rateBuf, inOutCtx.rate);
		oa::memcpy(outData, rateBuf + inOutCtx.bufLen, chunk);
		inOutCtx.bufLen += static_cast<oa::U32>(chunk);
		offset += chunk;
		if (inOutCtx.bufLen == inOutCtx.rate) {
			oa::keccakF1600(inOutCtx.state);
			inOutCtx.bufLen = 0;
		}
	}

	while (offset < inOutLen) {
		oa::Usize remaining = inOutLen - offset;
		if (remaining >= inOutCtx.rate) {
			extractBytesFromState(inOutCtx.state, outData + offset, inOutCtx.rate);
			offset += inOutCtx.rate;
			oa::keccakF1600(inOutCtx.state);
		} else {
			extractBytesFromState(inOutCtx.state, outData + offset, remaining);
			inOutCtx.bufLen = static_cast<oa::U32>(remaining);
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
static oa::Usize leftEncode(oa::U64 inVal, oa::Byte* outBuf) {
	oa::Byte tmp[9];
	oa::U32 n = 0;
	if (inVal == 0) {
		tmp[0] = 0;
		n = 1;
	} else {
		oa::U64 v = inVal;
		while (v > 0) {
			tmp[n++] = static_cast<oa::Byte>(v & 0xFF);
			v >>= 8;
		}
		// Reverse tmp[0..n-1] so it's big-endian
		for (oa::U32 i = 0; i < n / 2; ++i) {
			oa::Byte t = tmp[i]; tmp[i] = tmp[n - 1 - i]; tmp[n - 1 - i] = t;
		}
	}
	outBuf[0] = static_cast<oa::Byte>(n);
	oa::memcpy(outBuf + 1, tmp, n);
	return 1 + n;
}

// right_encode: encode integer as big-endian bytes suffixed by byte count.
static oa::Usize rightEncode(oa::U64 inVal, oa::Byte* outBuf) {
	oa::Byte tmp[9];
	oa::U32 n = 0;
	if (inVal == 0) {
		tmp[0] = 0;
		n = 1;
	} else {
		oa::U64 v = inVal;
		while (v > 0) {
			tmp[n++] = static_cast<oa::Byte>(v & 0xFF);
			v >>= 8;
		}
		for (oa::U32 i = 0; i < n / 2; ++i) {
			oa::Byte t = tmp[i]; tmp[i] = tmp[n - 1 - i]; tmp[n - 1 - i] = t;
		}
	}
	oa::memcpy(outBuf, tmp, n);
	outBuf[n] = static_cast<oa::Byte>(n);
	return n + 1;
}

// encode_string(S) = left_encode(len(S)*8) || S
static oa::Usize encodeString(const oa::Byte* inStr, oa::Usize inLen, oa::Byte* outBuf) {
	oa::Usize hLen = leftEncode(inLen * 8, outBuf);
	if (inLen != 0) {
		oa::memcpy(outBuf + hLen, inStr, inLen);
	}
	return hLen + inLen;
}

// bytepad(X, w): left_encode(w) || X || 0*pad
// We absorb the prefix directly into the sponge context.

oa::Status kmac256(
	const oa::Byte* inKey, oa::Usize inKeyLen,
	const oa::Byte* inData, oa::Usize inDataLen,
	const oa::Byte* inCustom, oa::Usize inCustomLen,
	oa::Byte* outMac, oa::Usize inOutLen)
{
	const oa::U32 rate = 136;
	constexpr oa::Usize kMax = oa::Limits<oa::Usize>::max();
	if ((inKey == nullptr && inKeyLen != 0) || (inData == nullptr && inDataLen != 0) ||
		(inCustom == nullptr && inCustomLen != 0) || (outMac == nullptr && inOutLen != 0)) {
		return oa::Status::invalidArgument("KMAC pointer is null with a non-zero length");
	}
	if (inKeyLen > oa::Limits<oa::U64>::max() / 8 ||
		inCustomLen > oa::Limits<oa::U64>::max() / 8 ||
		inOutLen > oa::Limits<oa::U64>::max() / 8) {
		return oa::Status::error(oa::StatusCode::OutOfRange,
			"KMAC bit length exceeds SP 800-185 encoding");
	}
	auto paddedSize = [](oa::Usize inSize) -> oa::Result<oa::Usize> {
		if (inSize > oa::Limits<oa::Usize>::max() - (rate - 1)) {
			return oa::Status::error(oa::StatusCode::OutOfRange, "KMAC encoded input is too large");
		}
		return ((inSize + rate - 1) / rate) * rate;
	};
	{
		oa::U64 state[25] = {};
		SecureEraseGuard stateGuard(state, sizeof(state));

		// cSHAKE prefix with function name N="KMAC" and customization S.
		if (inCustomLen > kMax - 32) {
				return oa::Status::error(oa::StatusCode::OutOfRange,
					"KMAC customization string is too large");
		}
		oa::Byte header[16];
		oa::Usize headerLen = leftEncode(rate, header);
		const oa::Byte kmacLabel[] = {'K', 'M', 'A', 'C'};
		headerLen += encodeString(kmacLabel, 4, header + headerLen);
		oa::Byte customLenEncoding[9];
		const oa::Usize customHeaderLen = leftEncode(inCustomLen * 8, customLenEncoding);
		const oa::Usize prefixRawLen = headerLen + customHeaderLen + inCustomLen;
		auto prefixSize = paddedSize(prefixRawLen);
		if (!prefixSize) return prefixSize.getStatus();
		auto prefixResult = SecureByteBuffer::create(prefixSize.getValue());
		if (!prefixResult) return prefixResult.getStatus();
		SecureByteBuffer prefix = oa::move(*prefixResult);
		oa::Usize pLen = 0;

		// left_encode(rate) for bytepad header
		pLen += leftEncode(rate, prefix.data() + pLen);

		// encode_string(N) where N = "KMAC"
		pLen += encodeString(kmacLabel, 4, prefix.data() + pLen);

		// encode_string(S) where S = inCustom
		pLen += encodeString(inCustom, inCustomLen, prefix.data() + pLen);
		pLen = prefix.size();

		// Absorb cSHAKE prefix
		oa::Usize offset = 0;
		while (offset + rate <= pLen) {
			xorBytesIntoState(state, prefix.data() + offset, rate);
			oa::keccakF1600(state);
			offset += rate;
		}
		if (offset < pLen) {
			xorBytesIntoState(state, prefix.data() + offset, pLen - offset);
		}

		// bytepad(encode_string(K), rate) for KMAC key
		if (inKeyLen > kMax - 20) {
			return oa::Status::error(oa::StatusCode::OutOfRange, "KMAC key is too large");
		}
		oa::Byte keyLenEncoding[9];
		const oa::Usize keyHeaderLen = leftEncode(inKeyLen * 8, keyLenEncoding);
		auto keySize = paddedSize(leftEncode(rate, header) + keyHeaderLen + inKeyLen);
		if (!keySize) return keySize.getStatus();
		auto keyResult = SecureByteBuffer::create(keySize.getValue());
		if (!keyResult) return keyResult.getStatus();
		SecureByteBuffer keyBuf = oa::move(*keyResult);
		oa::Usize kLen = 0;
		kLen += leftEncode(rate, keyBuf.data() + kLen);
		kLen += encodeString(inKey, inKeyLen, keyBuf.data() + kLen);
		kLen = keyBuf.size();

		// Absorb key block
		offset = 0;
		while (offset + rate <= kLen) {
			xorBytesIntoState(state, keyBuf.data() + offset, rate);
			oa::keccakF1600(state);
			offset += rate;
		}
		if (offset < kLen) {
			xorBytesIntoState(state, keyBuf.data() + offset, kLen - offset);
		}

		// Absorb message
		offset = 0;
		while (offset + rate <= inDataLen) {
			xorBytesIntoState(state, inData + offset, rate);
			oa::keccakF1600(state);
			offset += rate;
		}

		// Final block: remaining message + right_encode(outLen*8) + domain byte
		oa::Byte tail[512];
		SecureEraseGuard tailGuard(tail, sizeof(tail));
		oa::Usize tLen = inDataLen - offset;
		if (tLen != 0) {
			oa::memcpy(tail, inData + offset, tLen);
		}
		tLen += rightEncode(inOutLen * 8, tail + tLen);

		// cSHAKE domain: 0x04 (not SHAKE's 0x1F)
		tail[tLen] = 0x04;
		oa::memzero(tail + tLen + 1, rate - tLen - 1);
		tail[rate - 1] |= 0x80;
		xorBytesIntoState(state, tail, rate);
		oa::keccakF1600(state);

		// squeeze
		oa::Usize squeezed = 0;
		while (squeezed < inOutLen) {
			oa::Usize chunk = oa::min(inOutLen - squeezed, static_cast<oa::Usize>(rate));
			extractBytesFromState(state, outMac + squeezed, chunk);
			squeezed += chunk;
			if (squeezed < inOutLen) {
				oa::keccakF1600(state);
			}
		}
		return oa::Status::ok();
	}
}

} // namespace oa
