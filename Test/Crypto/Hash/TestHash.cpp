// OaHash/OaHasher/Merkle contract tests. These are pure CPU tests and cover
// malformed-input behavior in addition to the primitive KATs in TestKeccak.

#include <Oa/Crypto/Hash.h>
#include <Oa/Crypto/Keccak.h>

#include <gtest/gtest.h>

#include <cstring>

namespace {

OaHash HashByte(OaByte InValue) {
	OaHash out;
	OaShake256(&InValue, 1, out.Bytes.data(), out.Bytes.size());
	return out;
}

} // namespace

TEST(Hash, StrictHexRoundTrip) {
	const char* lower = "00112233445566778899aabbccddeeff"
		"fedcba98765432100123456789abcdef";
	auto parsed = OaHash::FromHex(lower);
	ASSERT_TRUE(parsed.IsOk());
	EXPECT_EQ(parsed.GetValue().ToHex(), lower);

	auto upper = OaHash::FromHex(
		"00112233445566778899AABBCCDDEEFF"
		"FEDCBA98765432100123456789ABCDEF");
	ASSERT_TRUE(upper.IsOk());
	EXPECT_EQ(upper.GetValue(), parsed.GetValue());
}

TEST(Hash, StrictHexRejectsMalformedInput) {
	EXPECT_TRUE(OaHash::FromHex("").IsError());
	EXPECT_TRUE(OaHash::FromHex(
		"00112233445566778899aabbccddeeff"
		"fedcba98765432100123456789abcdeg").IsError());
}

TEST(Hash, FromBytesRejectsWrongLength) {
	OaByte exact[32]{};
	OaByte shortData[31]{};
	EXPECT_TRUE(OaHash::FromBytes(exact).IsOk());
	EXPECT_TRUE(OaHash::FromBytes(shortData).IsError());
	EXPECT_TRUE(OaHash::FromBytes(OaSpan<const OaByte>(nullptr, 32)).IsError());
}

TEST(Hasher, FinalizeIsIdempotentAndRequiresResetBeforeUpdate) {
	const OaByte data[] = {'a', 'b', 'c'};
	OaHasher hasher;
	ASSERT_TRUE(hasher.Update(data, sizeof(data)).IsOk());
	const OaHash first = hasher.Finalize();
	EXPECT_EQ(hasher.Finalize(), first);
	EXPECT_TRUE(hasher.Update(data, sizeof(data)).IsError());

	hasher.Reset();
	ASSERT_TRUE(hasher.Update(data, sizeof(data)).IsOk());
	EXPECT_EQ(hasher.Finalize(), first);
}

TEST(Hasher, RejectsNullNonEmptyInput) {
	OaHasher hasher;
	EXPECT_TRUE(hasher.Update(nullptr, 1).IsError());
	EXPECT_TRUE(hasher.Update(nullptr, 0).IsOk());
}

TEST(Merkle, EmptyAndSingleLeafContracts) {
	OaVec<OaHash> empty;
	EXPECT_TRUE(OaMerkleRoot(empty).IsZero());
	auto emptyTree = OaBuildMerkleTree(empty);
	EXPECT_TRUE(emptyTree.Root.IsZero());
	EXPECT_TRUE(OaGetMerkleProof(emptyTree, 0).IsError());

	OaVec<OaHash> one{HashByte(7)};
	auto tree = OaBuildMerkleTree(one);
	EXPECT_EQ(tree.Root, one[0]);
	auto proof = OaGetMerkleProof(tree, 0);
	ASSERT_TRUE(proof.IsOk());
	EXPECT_TRUE(proof.GetValue().Siblings.Empty());
	EXPECT_TRUE(OaVerifyMerkleProof(one[0], proof.GetValue(), tree.Root));
}

TEST(Merkle, EveryLeafProofVerifiesForOddTree) {
	OaVec<OaHash> leaves;
	for (OaByte i = 0; i < 7; ++i) leaves.PushBack(HashByte(i));
	const auto tree = OaBuildMerkleTree(leaves);
	EXPECT_EQ(tree.Root, OaMerkleRoot(leaves));
	for (OaU32 i = 0; i < leaves.Size(); ++i) {
		auto proof = OaGetMerkleProof(tree, i);
		ASSERT_TRUE(proof.IsOk());
		EXPECT_TRUE(OaVerifyMerkleProof(leaves[i], proof.GetValue(), tree.Root));
		EXPECT_FALSE(OaVerifyMerkleProof(HashByte(99), proof.GetValue(), tree.Root));
	}
	EXPECT_TRUE(OaGetMerkleProof(tree, static_cast<OaU32>(leaves.Size())).IsError());
}

TEST(Merkle, MalformedDirectionVectorFailsClosed) {
	OaVec<OaHash> leaves{HashByte(1), HashByte(2)};
	const auto tree = OaBuildMerkleTree(leaves);
	auto proof = OaGetMerkleProof(tree, 0);
	ASSERT_TRUE(proof.IsOk());
	proof.GetValue().IsLeft.Clear();
	EXPECT_FALSE(OaVerifyMerkleProof(leaves[0], proof.GetValue(), tree.Root));
}
