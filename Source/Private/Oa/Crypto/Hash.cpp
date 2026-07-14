// OaHash CPU primitives: SHAKE-256 hashing, incremental hasher, Merkle tree.
// Zero external dependencies — uses Keccak.h only.

#include <Oa/Crypto/Hash.h>
#include <Oa/Crypto/Keccak.h>

// SHAKE-256(left || right) — used by the Merkle tree.
OaHash OaHashCombine(const OaHash& InLeft, const OaHash& InRight) {
	OaByte Buf[64];
	for (OaUsize i = 0; i < 32; ++i) { Buf[i] = InLeft.Bytes[i]; }
	for (OaUsize i = 0; i < 32; ++i) { Buf[32 + i] = InRight.Bytes[i]; }
	OaHash Hash;
	OaShake256(Buf, 64, Hash.Bytes.data(), 32);
	return Hash;
}

// OaHasher — incremental SHAKE-256
struct OaHasher::Impl {
	OaShakeCtx Ctx;
	OaHash Digest;
	OaBool Finalized = false;
};

OaHasher::OaHasher() : Impl_(OaMakeUniquePtr<Impl>()) {
	OaShake256Init(Impl_->Ctx);
}

OaHasher::~OaHasher() = default;

OaStatus OaHasher::Update(const OaByte* InData, OaUsize InLen) {
	if (Impl_->Finalized) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition,
			"OaHasher cannot absorb after Finalize; call Reset first");
	}
	if (InData == nullptr && InLen != 0) {
		return OaStatus::InvalidArgument("OaHasher input is null with a non-zero length");
	}
	OaShakeAbsorb(Impl_->Ctx, InData, InLen);
	return OaStatus::Ok();
}

OaHash OaHasher::Finalize() {
	if (!Impl_->Finalized) {
		OaShakeSqueeze(Impl_->Ctx, Impl_->Digest.Bytes.data(), 32);
		Impl_->Finalized = true;
	}
	return Impl_->Digest;
}

void OaHasher::Reset() {
	OaShake256Init(Impl_->Ctx);
	Impl_->Digest = OaHash::Zero();
	Impl_->Finalized = false;
}

// Merkle tree

OaHash OaMerkleRoot(const OaVec<OaHash>& InLeaves) {
	if (InLeaves.Empty()) {
		return OaHash::Zero();
	}
	if (InLeaves.Size() == 1) {
		return InLeaves[0];
	}

	OaVec<OaHash> Level = InLeaves;

	while (Level.Size() > 1) {
		OaVec<OaHash> NextLevel;
		NextLevel.Reserve((Level.Size() + 1) / 2);

		for (OaUsize i = 0; i < Level.Size(); i += 2) {
			if (i + 1 < Level.Size()) {
				NextLevel.PushBack(OaHashCombine(Level[i], Level[i + 1]));
			} else {
				NextLevel.PushBack(OaHashCombine(Level[i], Level[i]));
			}
		}

		Level = std::move(NextLevel);
	}

	return Level[0];
}

OaMerkleTree OaBuildMerkleTree(const OaVec<OaHash>& InLeaves) {
	OaMerkleTree Tree;

	if (InLeaves.Empty()) {
		Tree.Root = OaHash::Zero();
		return Tree;
	}

	Tree.Levels.PushBack(InLeaves);

	while (Tree.Levels.Back().Size() > 1) {
		const auto& PrevLevel = Tree.Levels.Back();
		OaVec<OaHash> NextLevel;
		NextLevel.Reserve((PrevLevel.Size() + 1) / 2);

		for (OaUsize i = 0; i < PrevLevel.Size(); i += 2) {
			if (i + 1 < PrevLevel.Size()) {
				NextLevel.PushBack(OaHashCombine(PrevLevel[i], PrevLevel[i + 1]));
			} else {
				NextLevel.PushBack(OaHashCombine(PrevLevel[i], PrevLevel[i]));
			}
		}

		Tree.Levels.PushBack(std::move(NextLevel));
	}

	Tree.Root = Tree.Levels.Back()[0];
	return Tree;
}

OaResult<OaMerkleProof> OaGetMerkleProof(
	const OaMerkleTree& InTree, OaU32 InLeafIndex) {
	OaMerkleProof Proof;

	if (InTree.Levels.Empty()) {
		return OaStatus::InvalidArgument("Cannot build a Merkle proof from an empty tree");
	}
	if (InLeafIndex >= InTree.Levels[0].Size()) {
		return OaStatus::Error(OaStatusCode::OutOfRange, "Merkle leaf index is out of range");
	}

	OaU32 Idx = InLeafIndex;

	for (OaUsize Level = 0; Level < InTree.Levels.Size() - 1; ++Level) {
		const auto& Nodes = InTree.Levels[Level];
		OaU32 SiblingIdx = (Idx % 2 == 0) ? Idx + 1 : Idx - 1;

		if (SiblingIdx < Nodes.Size()) {
			Proof.Siblings.PushBack(Nodes[SiblingIdx]);
		} else {
			Proof.Siblings.PushBack(Nodes[Idx]);
		}

		Proof.IsLeft.PushBack(Idx % 2 == 1);
		Idx /= 2;
	}

	return Proof;
}

OaBool OaVerifyMerkleProof(
	const OaHash& InLeaf,
	const OaMerkleProof& InProof,
	const OaHash& InRoot
) {
	if (InProof.Siblings.Size() != InProof.IsLeft.Size()) {
		return false;
	}
	OaHash Current = InLeaf;

	for (OaUsize i = 0; i < InProof.Siblings.Size(); ++i) {
		if (InProof.IsLeft[i]) {
			Current = OaHashCombine(InProof.Siblings[i], Current);
		} else {
			Current = OaHashCombine(Current, InProof.Siblings[i]);
		}
	}

	return Current == InRoot;
}
