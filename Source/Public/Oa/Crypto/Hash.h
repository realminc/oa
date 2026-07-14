// OaHash — 32-byte SHAKE-256 hash, incremental hasher, Merkle tree.
//
// Domain-specific aliases (OaAddress, OaBlockHash, OaTxHash, OaStateRoot,
// OaValidatorId) and any blockchain bookkeeping belong in a separate module
// (see Code/GitHub/chain). This header is the pure crypto core.

#pragma once

#include <cstring>

#include <Oa/Core.h>

// OaHash — 32-byte cryptographic hash (SHAKE-256 output)
class OaHash {
public:
	OaArray<OaByte, 32> Bytes;

	constexpr OaHash() : Bytes{} {}

	[[nodiscard]] static constexpr OaHash Zero() { return OaHash{}; }

	[[nodiscard]] static OaResult<OaHash> FromBytes(OaSpan<const OaByte> InBytes) {
		if (InBytes.Size() != 32 || InBytes.Data() == nullptr) {
			return OaStatus::InvalidArgument("OaHash requires exactly 32 bytes");
		}
		OaHash Hash;
		std::memcpy(Hash.Bytes.data(), InBytes.Data(), Hash.Bytes.size());
		return Hash;
	}

	[[nodiscard]] static OaResult<OaHash> FromHex(OaStringView InHex) {
		OaHash Hash;
		if (InHex.size() != 64) {
			return OaStatus::InvalidArgument("OaHash hex text must contain exactly 64 characters");
		}

		auto HexVal = [](char c) -> OaI32 {
			if (c >= '0' && c <= '9') { return static_cast<OaByte>(c - '0'); }
			if (c >= 'a' && c <= 'f') { return static_cast<OaByte>(10 + c - 'a'); }
			if (c >= 'A' && c <= 'F') { return static_cast<OaByte>(10 + c - 'A'); }
			return -1;
		};

		for (OaUsize i = 0; i < 32; ++i) {
			const OaI32 hi = HexVal(InHex[i * 2]);
			const OaI32 lo = HexVal(InHex[i * 2 + 1]);
			if (hi < 0 || lo < 0) {
				return OaStatus::InvalidArgument("OaHash hex text contains a non-hexadecimal character");
			}
			Hash.Bytes[i] = static_cast<OaByte>(
				(static_cast<OaU32>(hi) << 4) | static_cast<OaU32>(lo));
		}
		return Hash;
	}

	[[nodiscard]] bool operator==(const OaHash& InOther) const {
		return Bytes == InOther.Bytes;
	}

	[[nodiscard]] bool operator!=(const OaHash& InOther) const {
		return Bytes != InOther.Bytes;
	}

	[[nodiscard]] bool operator<(const OaHash& InOther) const {
		return Bytes < InOther.Bytes;
	}

	[[nodiscard]] bool IsZero() const {
		for (auto B : Bytes) {
			if (B != 0) {
				return false;
			}
		}
		return true;
	}

	[[nodiscard]] OaString ToHex() const {
		static const char Hex[] = "0123456789abcdef";
		OaString Result;
		Result.reserve(64);
		for (auto B : Bytes) {
			Result += Hex[B >> 4];
			Result += Hex[B & 0xF];
		}
		return Result;
	}

	[[nodiscard]] OaString ToShortHex() const { return ToHex().substr(0, 16); }

	[[nodiscard]] const OaByte* Data() const { return Bytes.data(); }
	[[nodiscard]] OaByte* Data() { return Bytes.data(); }

	[[nodiscard]] static constexpr OaUsize Size() { return 32; }
};

// std::unordered_map/set hasher
class OaHashHasher {
public:
	[[nodiscard]] OaUsize operator()(const OaHash& InHash) const {
		OaUsize Result = sizeof(OaUsize) == 8
			? static_cast<OaUsize>(1469598103934665603ULL)
			: static_cast<OaUsize>(2166136261U);
		const OaUsize Prime = sizeof(OaUsize) == 8
			? static_cast<OaUsize>(1099511628211ULL)
			: static_cast<OaUsize>(16777619U);
		for (OaByte Byte : InHash.Bytes) {
			Result ^= Byte;
			Result *= Prime;
		}
		return Result;
	}
};

// Internal helper: SHAKE-256(left || right) → 32 bytes.
// Used by OaMerkleRoot/OaMerkleTree; available for unit tests.
OaHash OaHashCombine(const OaHash& InLeft, const OaHash& InRight);

// Incremental hasher (SHAKE-256)
class OaHasher {
public:
	OaHasher();
	~OaHasher();

	[[nodiscard]] OaStatus Update(const OaByte* InData, OaUsize InLen);
	OaHash Finalize();
	void Reset();

private:
	struct Impl;
	OaUniquePtr<Impl> Impl_;
};

// Merkle tree (CPU). For GPU batch trees, use OaFnHash::MerkleRoot.

OaHash OaMerkleRoot(const OaVec<OaHash>& InLeaves);

class OaMerkleTree {
public:
	OaVec<OaVec<OaHash>> Levels;
	OaHash Root;
};

OaMerkleTree OaBuildMerkleTree(const OaVec<OaHash>& InLeaves);

class OaMerkleProof {
public:
	OaVec<OaHash> Siblings;
	OaVec<OaBool> IsLeft;
};

[[nodiscard]] OaResult<OaMerkleProof> OaGetMerkleProof(
	const OaMerkleTree& InTree, OaU32 InLeafIndex);

OaBool OaVerifyMerkleProof(
	const OaHash& InLeaf,
	const OaMerkleProof& InProof,
	const OaHash& InRoot);
