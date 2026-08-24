// oa::Hash — 32-byte SHAKE-256 hash, incremental hasher, Merkle tree.
//
// Domain-specific aliases and any blockchain bookkeeping belong in a separate
// module. This header is the pure crypto core.

#pragma once

#include <cstring>

#include <oa/core.h>

namespace oa {

// Hash — 32-byte cryptographic hash (SHAKE-256 output)
class Hash {
public:
	oa::Array<oa::Byte, 32> bytes;

	constexpr Hash() : bytes{} {}

	[[nodiscard]] static constexpr Hash zero() { return Hash{}; }

	[[nodiscard]] static oa::Result<Hash> fromBytes(oa::Span<const oa::Byte> inBytes) {
		if (inBytes.size() != 32 || inBytes.data() == nullptr) {
			return oa::Status::invalidArgument("oa::Hash requires exactly 32 bytes");
		}
		Hash hash;
		std::memcpy(hash.bytes.data(), inBytes.data(), hash.bytes.size());
		return hash;
	}

	[[nodiscard]] static oa::Result<Hash> fromHex(oa::StringView inHex) {
		Hash hash;
		if (inHex.size() != 64) {
			return oa::Status::invalidArgument("oa::Hash hex text must contain exactly 64 characters");
		}

		auto hexVal = [](char c) -> oa::I32 {
			if (c >= '0' && c <= '9') { return static_cast<oa::Byte>(c - '0'); }
			if (c >= 'a' && c <= 'f') { return static_cast<oa::Byte>(10 + c - 'a'); }
			if (c >= 'A' && c <= 'F') { return static_cast<oa::Byte>(10 + c - 'A'); }
			return -1;
		};

		for (oa::Usize i = 0; i < 32; ++i) {
			const oa::I32 hi = hexVal(inHex[i * 2]);
			const oa::I32 lo = hexVal(inHex[i * 2 + 1]);
			if (hi < 0 || lo < 0) {
				return oa::Status::invalidArgument("oa::Hash hex text contains a non-hexadecimal character");
			}
			hash.bytes[i] = static_cast<oa::Byte>(
				(static_cast<oa::U32>(hi) << 4) | static_cast<oa::U32>(lo));
		}
		return hash;
	}

	[[nodiscard]] bool operator==(const Hash& inOther) const {
		return bytes == inOther.bytes;
	}

	[[nodiscard]] bool operator!=(const Hash& inOther) const {
		return bytes != inOther.bytes;
	}

	[[nodiscard]] bool operator<(const Hash& inOther) const {
		return bytes < inOther.bytes;
	}

	[[nodiscard]] bool isZero() const {
		for (auto b : bytes) {
			if (b != 0) {
				return false;
			}
		}
		return true;
	}

	[[nodiscard]] oa::String toHex() const {
		static const char kHex[] = "0123456789abcdef";
		oa::String result;
		result.reserve(64);
		for (auto b : bytes) {
			result += kHex[b >> 4];
			result += kHex[b & 0xF];
		}
		return result;
	}

	[[nodiscard]] oa::String toShortHex() const { return toHex().substr(0, 16); }

	[[nodiscard]] const oa::Byte* data() const { return bytes.data(); }
	[[nodiscard]] oa::Byte* data() { return bytes.data(); }

	[[nodiscard]] static constexpr oa::Usize size() { return 32; }
};

// std::unordered_map/set hasher
class HashHasher {
public:
	[[nodiscard]] oa::Usize operator()(const Hash& inHash) const {
		oa::Usize result = sizeof(oa::Usize) == 8
			? static_cast<oa::Usize>(1469598103934665603ULL)
			: static_cast<oa::Usize>(2166136261U);
		const oa::Usize prime = sizeof(oa::Usize) == 8
			? static_cast<oa::Usize>(1099511628211ULL)
			: static_cast<oa::Usize>(16777619U);
		for (oa::Byte byte : inHash.bytes) {
			result ^= byte;
			result *= prime;
		}
		return result;
	}
};

// Internal helper: SHAKE-256(left || right) → 32 bytes.
// Used by merkleRoot/MerkleTree; available for unit tests.
Hash hashCombine(const Hash& inLeft, const Hash& inRight);

// Incremental hasher (SHAKE-256)
class Hasher {
public:
	Hasher();
	~Hasher();

	[[nodiscard]] oa::Status update(const oa::Byte* inData, oa::Usize inLen);
	Hash finalize();
	void reset();

private:
	struct Impl;
	oa::UniquePtr<Impl> impl_;
};

// Merkle tree (CPU). For GPU batch trees, use oa::FnHash::merkleRoot.

Hash merkleRoot(const oa::Vec<Hash>& inLeaves);

class MerkleTree {
public:
	oa::Vec<oa::Vec<Hash>> levels;
	Hash root;
};

MerkleTree buildMerkleTree(const oa::Vec<Hash>& inLeaves);

class MerkleProof {
public:
	oa::Vec<Hash> siblings;
	oa::Vec<oa::Bool> isLeft;
};

[[nodiscard]] oa::Result<MerkleProof> getMerkleProof(const MerkleTree& inTree, oa::U32 inLeafIndex);

oa::Bool verifyMerkleProof(
	const Hash& inLeaf,
	const MerkleProof& inProof,
	const Hash& inRoot);

} // namespace oa
