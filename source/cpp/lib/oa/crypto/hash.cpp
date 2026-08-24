// Hash CPU primitives: SHAKE-256 hashing, incremental hasher, Merkle tree.
// Zero external dependencies — uses Keccak.h only.

#include <oa/crypto/hash.h>
#include <oa/crypto/keccak.h>

namespace oa {

// SHAKE-256(left || right) — used by the Merkle tree.
Hash hashCombine(const Hash& inLeft, const Hash& inRight) {
	oa::Byte buf[64];
	for (oa::Usize i = 0; i < 32; ++i) { buf[i] = inLeft.bytes[i]; }
	for (oa::Usize i = 0; i < 32; ++i) { buf[32 + i] = inRight.bytes[i]; }
	Hash hash;
	::oa::shake256(buf, 64, hash.bytes.data(), 32);
	return hash;
}

// Hasher — incremental SHAKE-256
struct Hasher::Impl {
	::oa::ShakeContext ctx;
	Hash digest;
	oa::Bool finalized = false;
};

Hasher::Hasher() : impl_(oa::makeUnique<Impl>()) {
	::oa::shake256Init(impl_->ctx);
}

Hasher::~Hasher() = default;

oa::Status Hasher::update(const oa::Byte* inData, oa::Usize inLen) {
	if (impl_->finalized) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"oa::Hasher cannot absorb after finalize; call reset first");
	}
	if (inData == nullptr && inLen != 0) {
		return oa::Status::invalidArgument("oa::Hasher input is null with a non-zero length");
	}
	::oa::shakeAbsorb(impl_->ctx, inData, inLen);
	return oa::Status::ok();
}

Hash Hasher::finalize() {
	if (!impl_->finalized) {
		::oa::shakeSqueeze(impl_->ctx, impl_->digest.bytes.data(), 32);
		impl_->finalized = true;
	}
	return impl_->digest;
}

void Hasher::reset() {
	::oa::shake256Init(impl_->ctx);
	impl_->digest = Hash::zero();
	impl_->finalized = false;
}

// Merkle tree

Hash merkleRoot(const oa::Vec<Hash>& inLeaves) {
	if (inLeaves.empty()) {
		return Hash::zero();
	}
	if (inLeaves.size() == 1) {
		return inLeaves[0];
	}

	oa::Vec<Hash> level = inLeaves;

	while (level.size() > 1) {
		oa::Vec<Hash> nextLevel;
		nextLevel.reserve((level.size() + 1) / 2);

		for (oa::Usize i = 0; i < level.size(); i += 2) {
			if (i + 1 < level.size()) {
				nextLevel.pushBack(hashCombine(level[i], level[i + 1]));
			} else {
				nextLevel.pushBack(hashCombine(level[i], level[i]));
			}
		}

		level = oa::move(nextLevel);
	}

	return level[0];
}

MerkleTree buildMerkleTree(const oa::Vec<Hash>& inLeaves) {
	MerkleTree tree;

	if (inLeaves.empty()) {
		tree.root = Hash::zero();
		return tree;
	}

	tree.levels.pushBack(inLeaves);

	while (tree.levels.back().size() > 1) {
		const auto& previousLevel = tree.levels.back();
		oa::Vec<Hash> nextLevel;
		nextLevel.reserve((previousLevel.size() + 1) / 2);

		for (oa::Usize i = 0; i < previousLevel.size(); i += 2) {
			if (i + 1 < previousLevel.size()) {
				nextLevel.pushBack(hashCombine(previousLevel[i], previousLevel[i + 1]));
			} else {
				nextLevel.pushBack(hashCombine(previousLevel[i], previousLevel[i]));
			}
		}

		tree.levels.pushBack(oa::move(nextLevel));
	}

	tree.root = tree.levels.back()[0];
	return tree;
}

oa::Result<MerkleProof> getMerkleProof(
	const MerkleTree& inTree, oa::U32 inLeafIndex) {
	MerkleProof proof;

	if (inTree.levels.empty()) {
		return oa::Status::invalidArgument("Cannot build a Merkle proof from an empty tree");
	}
	if (inLeafIndex >= inTree.levels[0].size()) {
		return oa::Status::error(oa::StatusCode::OutOfRange, "Merkle leaf index is out of range");
	}

	oa::U32 idx = inLeafIndex;

	for (oa::Usize level = 0; level < inTree.levels.size() - 1; ++level) {
		const auto& nodes = inTree.levels[level];
		oa::U32 SiblingIdx = (idx % 2 == 0) ? idx + 1 : idx - 1;

		if (SiblingIdx < nodes.size()) {
			proof.siblings.pushBack(nodes[SiblingIdx]);
		} else {
			proof.siblings.pushBack(nodes[idx]);
		}

		proof.isLeft.pushBack(idx % 2 == 1);
		idx /= 2;
	}

	return proof;
}

oa::Bool verifyMerkleProof(
	const Hash& inLeaf,
	const MerkleProof& inProof,
	const Hash& inRoot
) {
	if (inProof.siblings.size() != inProof.isLeft.size()) {
		return false;
	}
	Hash current = inLeaf;

	for (oa::Usize i = 0; i < inProof.siblings.size(); ++i) {
		if (inProof.isLeft[i]) {
			current = hashCombine(inProof.siblings[i], current);
		} else {
			current = hashCombine(current, inProof.siblings[i]);
		}
	}

	return current == inRoot;
}

} // namespace oa
