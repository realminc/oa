// ML-DSA-65 digital signatures (standardized successor to Dilithium3).
// Post-quantum, NIST level 3 (~AES-192). No ECDSA. No BLS.
//
// Public key: 1952 bytes, secret key: 4032 bytes, Signature: 3309 bytes.
//
// Only single-shot keygen/sign/verify live here. The incomplete vulkan ML-DSA
// kernels are intentionally not exposed through a public batch API. Address
// derivation and blockchain bookkeeping belong in a downstream module.

#pragma once

#include <cstring>

#include <oa/core.h>
#include <oa/crypto/hash.h>

inline constexpr oa::Usize OA_SIGN_PUBKEY_SIZE = 1952;
inline constexpr oa::Usize OA_SIGN_SECRET_SIZE = 4032;
inline constexpr oa::Usize OA_SIGN_SIG_SIZE    = 3309;

namespace oa {

class PublicKey {
public:
	oa::Array<oa::Byte, OA_SIGN_PUBKEY_SIZE> bytes;

	constexpr PublicKey() : bytes{} {}

	[[nodiscard]] bool operator==(const PublicKey& inOther) const {
		return bytes == inOther.bytes;
	}

	[[nodiscard]] bool operator!=(const PublicKey& inOther) const {
		return bytes != inOther.bytes;
	}

	[[nodiscard]] bool isZero() const {
		for (auto b : bytes) {
			if (b != 0) { return false; }
		}
		return true;
	}

	[[nodiscard]] oa::String toShortHex() const {
		static const char kHex[] = "0123456789abcdef";
		oa::String result;
		result.reserve(32);
		for (oa::Usize i = 0; i < 16; ++i) {
			result += kHex[bytes[i] >> 4];
			result += kHex[bytes[i] & 0xF];
		}
		return result;
	}
};

class SecretKey {
public:
	oa::Array<oa::Byte, OA_SIGN_SECRET_SIZE> bytes;

	constexpr SecretKey() : bytes{} {}

	void secureZero() {
		volatile oa::Byte* ptr = bytes.data();
		for (oa::Usize i = 0; i < bytes.size(); ++i) {
			ptr[i] = 0;
		}
	}

	~SecretKey() { secureZero(); }

	SecretKey(const SecretKey&) = delete;
	SecretKey& operator=(const SecretKey&) = delete;

	SecretKey(SecretKey&& inOther) noexcept {
		std::memcpy(bytes.data(), inOther.bytes.data(), OA_SIGN_SECRET_SIZE);
		inOther.secureZero();
	}

	SecretKey& operator=(SecretKey&& inOther) noexcept {
		if (this != &inOther) {
			secureZero();
			std::memcpy(bytes.data(), inOther.bytes.data(), OA_SIGN_SECRET_SIZE);
			inOther.secureZero();
		}
		return *this;
	}
};

class Signature {
public:
	oa::Array<oa::Byte, OA_SIGN_SIG_SIZE> bytes;

	constexpr Signature() : bytes{} {}

	[[nodiscard]] bool operator==(const Signature& inOther) const {
		return bytes == inOther.bytes;
	}

	[[nodiscard]] bool isZero() const {
		for (auto b : bytes) {
			if (b != 0) { return false; }
		}
		return true;
	}

	[[nodiscard]] oa::String toShortHex() const {
		static const char kHex[] = "0123456789abcdef";
		oa::String result;
		result.reserve(32);
		for (oa::Usize i = 0; i < 16; ++i) {
			result += kHex[bytes[i] >> 4];
			result += kHex[bytes[i] & 0xF];
		}
		return result;
	}
};

class Keypair {
public:
	PublicKey pubkey;
	SecretKey secret;

	Keypair() = default;

	Keypair(Keypair&&) noexcept = default;
	Keypair& operator=(Keypair&&) noexcept = default;
	Keypair(const Keypair&) = delete;
	Keypair& operator=(const Keypair&) = delete;
};

// key generation (random). Deterministic seed-derived keygen is not exposed:
// liboqs 0.15 provides only random ML-DSA keypair generation (see sign.cpp).
[[nodiscard]] oa::Result<Keypair> generateKeypair();

// Sign a message
[[nodiscard]] oa::Result<Signature> sign(
	const oa::Byte* inMessage,
	oa::Usize inMessageLen,
	const SecretKey& inSecret);

// Sign a hash directly
[[nodiscard]] inline oa::Result<Signature> sign(
	const Hash& inHash,
	const SecretKey& inSecret) {
	return sign(inHash.bytes.data(), 32, inSecret);
}

// verify a signature
[[nodiscard]] oa::Bool verify(
	const oa::Byte* inMessage,
	oa::Usize inMessageLen,
	const Signature& inSignature,
	const PublicKey& inPubkey);

// verify a hash signature
[[nodiscard]] inline oa::Bool verify(
	const Hash& inHash,
	const Signature& inSignature,
	const PublicKey& inPubkey) {
	return verify(inHash.bytes.data(), 32, inSignature, inPubkey);
}

// Fixed-size serialization. Parsing is length-checked and never reads from a
// raw pointer without a caller-supplied extent.
[[nodiscard]] inline oa::Array<oa::Byte, OA_SIGN_PUBKEY_SIZE> serializePublicKey(
	const PublicKey& inKey) {
	return inKey.bytes;
}

[[nodiscard]] inline oa::Result<PublicKey> deserializePublicKey(
	oa::Span<const oa::Byte> inBuffer) {
	if (inBuffer.size() != OA_SIGN_PUBKEY_SIZE || inBuffer.data() == nullptr) {
		return oa::Status::invalidArgument("ML-DSA-65 public key must contain exactly 1952 bytes");
	}
	PublicKey key;
	std::memcpy(key.bytes.data(), inBuffer.data(), OA_SIGN_PUBKEY_SIZE);
	return key;
}

[[nodiscard]] inline oa::Array<oa::Byte, OA_SIGN_SIG_SIZE> serializeSignature(
	const Signature& inSig) {
	return inSig.bytes;
}

[[nodiscard]] inline oa::Result<Signature> deserializeSignature(
	oa::Span<const oa::Byte> inBuffer) {
	if (inBuffer.size() != OA_SIGN_SIG_SIZE || inBuffer.data() == nullptr) {
		return oa::Status::invalidArgument("ML-DSA-65 signature must contain exactly 3309 bytes");
	}
	Signature sig;
	std::memcpy(sig.bytes.data(), inBuffer.data(), OA_SIGN_SIG_SIZE);
	return sig;
}

class PublicKeyHasher {
public:
	[[nodiscard]] oa::Usize operator()(const PublicKey& inKey) const {
		oa::Usize result = sizeof(oa::Usize) == 8
			? static_cast<oa::Usize>(1469598103934665603ULL)
			: static_cast<oa::Usize>(2166136261U);
		const oa::Usize prime = sizeof(oa::Usize) == 8
			? static_cast<oa::Usize>(1099511628211ULL)
			: static_cast<oa::Usize>(16777619U);
		for (oa::Byte byte : inKey.bytes) {
			result ^= byte;
			result *= prime;
		}
		return result;
	}
};

} // namespace oa
