// ML-DSA-65 digital signatures (standardized successor to Dilithium3).
// Post-quantum, NIST Level 3 (~AES-192). No ECDSA. No BLS.
//
// Public key: 1952 bytes, Secret key: 4032 bytes, Signature: 3309 bytes.
//
// Only single-shot keygen/sign/verify live here. The incomplete Vulkan ML-DSA
// kernels are intentionally not exposed through a public batch API. Address
// derivation and blockchain bookkeeping belong in a downstream module.

#pragma once

#include <cstring>

#include <Oa/Core.h>
#include <Oa/Crypto/Hash.h>

inline constexpr OaUsize OA_SIGN_PUBKEY_SIZE = 1952;
inline constexpr OaUsize OA_SIGN_SECRET_SIZE = 4032;
inline constexpr OaUsize OA_SIGN_SIG_SIZE    = 3309;

class OaPublicKey {
public:
	OaArray<OaByte, OA_SIGN_PUBKEY_SIZE> Bytes;

	constexpr OaPublicKey() : Bytes{} {}

	[[nodiscard]] bool operator==(const OaPublicKey& InOther) const {
		return Bytes == InOther.Bytes;
	}

	[[nodiscard]] bool operator!=(const OaPublicKey& InOther) const {
		return Bytes != InOther.Bytes;
	}

	[[nodiscard]] bool IsZero() const {
		for (auto B : Bytes) {
			if (B != 0) { return false; }
		}
		return true;
	}

	[[nodiscard]] OaString ToShortHex() const {
		static const char Hex[] = "0123456789abcdef";
		OaString Result;
		Result.reserve(32);
		for (OaUsize i = 0; i < 16; ++i) {
			Result += Hex[Bytes[i] >> 4];
			Result += Hex[Bytes[i] & 0xF];
		}
		return Result;
	}
};

class OaSecretKey {
public:
	OaArray<OaByte, OA_SIGN_SECRET_SIZE> Bytes;

	constexpr OaSecretKey() : Bytes{} {}

	void SecureZero() {
		volatile OaByte* Ptr = Bytes.data();
		for (OaUsize i = 0; i < Bytes.size(); ++i) {
			Ptr[i] = 0;
		}
	}

	~OaSecretKey() { SecureZero(); }

	OaSecretKey(const OaSecretKey&) = delete;
	OaSecretKey& operator=(const OaSecretKey&) = delete;

	OaSecretKey(OaSecretKey&& InOther) noexcept {
		std::memcpy(Bytes.data(), InOther.Bytes.data(), OA_SIGN_SECRET_SIZE);
		InOther.SecureZero();
	}

	OaSecretKey& operator=(OaSecretKey&& InOther) noexcept {
		if (this != &InOther) {
			SecureZero();
			std::memcpy(Bytes.data(), InOther.Bytes.data(), OA_SIGN_SECRET_SIZE);
			InOther.SecureZero();
		}
		return *this;
	}
};

class OaSignature {
public:
	OaArray<OaByte, OA_SIGN_SIG_SIZE> Bytes;

	constexpr OaSignature() : Bytes{} {}

	[[nodiscard]] bool operator==(const OaSignature& InOther) const {
		return Bytes == InOther.Bytes;
	}

	[[nodiscard]] bool IsZero() const {
		for (auto B : Bytes) {
			if (B != 0) { return false; }
		}
		return true;
	}

	[[nodiscard]] OaString ToShortHex() const {
		static const char Hex[] = "0123456789abcdef";
		OaString Result;
		Result.reserve(32);
		for (OaUsize i = 0; i < 16; ++i) {
			Result += Hex[Bytes[i] >> 4];
			Result += Hex[Bytes[i] & 0xF];
		}
		return Result;
	}
};

class OaKeypair {
public:
	OaPublicKey Pubkey;
	OaSecretKey Secret;

	OaKeypair() = default;

	OaKeypair(OaKeypair&&) noexcept = default;
	OaKeypair& operator=(OaKeypair&&) noexcept = default;
	OaKeypair(const OaKeypair&) = delete;
	OaKeypair& operator=(const OaKeypair&) = delete;
};

// Key generation (random). Deterministic seed-derived keygen is not exposed:
// liboqs 0.15 provides only random ML-DSA keypair generation (see Sign.cpp).
[[nodiscard]] OaResult<OaKeypair> OaGenerateKeypair();

// Sign a message
[[nodiscard]] OaResult<OaSignature> OaSign(
	const OaByte* InMessage,
	OaUsize InMessageLen,
	const OaSecretKey& InSecret);

// Sign a hash directly
[[nodiscard]] inline OaResult<OaSignature> OaSign(
	const OaHash& InHash,
	const OaSecretKey& InSecret) {
	return OaSign(InHash.Bytes.data(), 32, InSecret);
}

// Verify a signature
[[nodiscard]] OaBool OaVerify(
	const OaByte* InMessage,
	OaUsize InMessageLen,
	const OaSignature& InSignature,
	const OaPublicKey& InPubkey);

// Verify a hash signature
[[nodiscard]] inline OaBool OaVerify(
	const OaHash& InHash,
	const OaSignature& InSignature,
	const OaPublicKey& InPubkey) {
	return OaVerify(InHash.Bytes.data(), 32, InSignature, InPubkey);
}

// Fixed-size serialization. Parsing is length-checked and never reads from a
// raw pointer without a caller-supplied extent.
[[nodiscard]] inline OaArray<OaByte, OA_SIGN_PUBKEY_SIZE> OaSerializePublicKey(
	const OaPublicKey& InKey) {
	return InKey.Bytes;
}

[[nodiscard]] inline OaResult<OaPublicKey> OaDeserializePublicKey(
	OaSpan<const OaByte> InBuffer) {
	if (InBuffer.Size() != OA_SIGN_PUBKEY_SIZE || InBuffer.Data() == nullptr) {
		return OaStatus::InvalidArgument("ML-DSA-65 public key must contain exactly 1952 bytes");
	}
	OaPublicKey Key;
	std::memcpy(Key.Bytes.data(), InBuffer.Data(), OA_SIGN_PUBKEY_SIZE);
	return Key;
}

[[nodiscard]] inline OaArray<OaByte, OA_SIGN_SIG_SIZE> OaSerializeSignature(
	const OaSignature& InSig) {
	return InSig.Bytes;
}

[[nodiscard]] inline OaResult<OaSignature> OaDeserializeSignature(
	OaSpan<const OaByte> InBuffer) {
	if (InBuffer.Size() != OA_SIGN_SIG_SIZE || InBuffer.Data() == nullptr) {
		return OaStatus::InvalidArgument("ML-DSA-65 signature must contain exactly 3309 bytes");
	}
	OaSignature Sig;
	std::memcpy(Sig.Bytes.data(), InBuffer.Data(), OA_SIGN_SIG_SIZE);
	return Sig;
}

class OaPublicKeyHasher {
public:
	[[nodiscard]] OaUsize operator()(const OaPublicKey& InKey) const {
		OaUsize Result = sizeof(OaUsize) == 8
			? static_cast<OaUsize>(1469598103934665603ULL)
			: static_cast<OaUsize>(2166136261U);
		const OaUsize Prime = sizeof(OaUsize) == 8
			? static_cast<OaUsize>(1099511628211ULL)
			: static_cast<OaUsize>(16777619U);
		for (OaByte Byte : InKey.Bytes) {
			Result ^= Byte;
			Result *= Prime;
		}
		return Result;
	}
};
