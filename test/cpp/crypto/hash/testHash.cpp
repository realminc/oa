// oa::Hash/oa::Hasher/Merkle contract tests. These are pure CPU tests and cover
// malformed-input behavior in addition to the primitive KATs in TestKeccak.

#include <oa/crypto/hash.h>
#include <oa/crypto/keccak.h>

#include <gtest/gtest.h>

#include <cstring>

namespace {

oa::Hash hashByte(oa::Byte inValue) {
	oa::Hash out;
	oa::shake256(&inValue, 1, out.bytes.data(), out.bytes.size());
	return out;
}

} // namespace

TEST(Hash, StrictHexRoundTrip) {
	const char* lower = "00112233445566778899aabbccddeeff"
		"fedcba98765432100123456789abcdef";
	auto parsed = oa::Hash::fromHex(lower);
	ASSERT_TRUE(parsed.isOk());
	EXPECT_EQ(parsed.getValue().toHex(), lower);

	auto upper = oa::Hash::fromHex(
		"00112233445566778899AABBCCDDEEFF"
		"FEDCBA98765432100123456789ABCDEF");
	ASSERT_TRUE(upper.isOk());
	EXPECT_EQ(upper.getValue(), parsed.getValue());
}

TEST(Hash, StrictHexRejectsMalformedInput) {
	EXPECT_TRUE(oa::Hash::fromHex("").isError());
	EXPECT_TRUE(oa::Hash::fromHex(
		"00112233445566778899aabbccddeeff"
		"fedcba98765432100123456789abcdeg").isError());
}

TEST(Hash, FromBytesRejectsWrongLength) {
	oa::Byte exact[32]{};
	oa::Byte shortData[31]{};
	EXPECT_TRUE(oa::Hash::fromBytes(exact).isOk());
	EXPECT_TRUE(oa::Hash::fromBytes(shortData).isError());
}

TEST(Hasher, FinalizeIsIdempotentAndRequiresResetBeforeUpdate) {
	const oa::Byte data[] = {'a', 'b', 'c'};
	oa::Hasher hasher;
	ASSERT_TRUE(hasher.update(data, sizeof(data)).isOk());
	const oa::Hash first = hasher.finalize();
	EXPECT_EQ(hasher.finalize(), first);
	EXPECT_TRUE(hasher.update(data, sizeof(data)).isError());

	hasher.reset();
	ASSERT_TRUE(hasher.update(data, sizeof(data)).isOk());
	EXPECT_EQ(hasher.finalize(), first);
}

TEST(Hasher, RejectsNullNonEmptyInput) {
	oa::Hasher hasher;
	EXPECT_TRUE(hasher.update(nullptr, 1).isError());
	EXPECT_TRUE(hasher.update(nullptr, 0).isOk());
}

TEST(Merkle, EmptyAndSingleLeafContracts) {
	oa::Vector<oa::Hash> empty;
	EXPECT_TRUE(oa::merkleRoot(empty).isZero());
	auto emptyTree = oa::buildMerkleTree(empty);
	EXPECT_TRUE(emptyTree.root.isZero());
	EXPECT_TRUE(oa::getMerkleProof(emptyTree, 0).isError());

	oa::Vector<oa::Hash> one{hashByte(7)};
	auto tree = oa::buildMerkleTree(one);
	EXPECT_EQ(tree.root, one[0]);
	auto proof = oa::getMerkleProof(tree, 0);
	ASSERT_TRUE(proof.isOk());
	EXPECT_TRUE(proof.getValue().siblings.empty());
	EXPECT_TRUE(oa::verifyMerkleProof(one[0], proof.getValue(), tree.root));
}

TEST(Merkle, EveryLeafProofVerifiesForOddTree) {
	oa::Vector<oa::Hash> leaves;
	for (oa::Byte i = 0; i < 7; ++i) leaves.pushBack(hashByte(i));
	const auto tree = oa::buildMerkleTree(leaves);
	EXPECT_EQ(tree.root, oa::merkleRoot(leaves));
	for (oa::U32 i = 0; i < leaves.size(); ++i) {
		auto proof = oa::getMerkleProof(tree, i);
		ASSERT_TRUE(proof.isOk());
		EXPECT_TRUE(oa::verifyMerkleProof(leaves[i], proof.getValue(), tree.root));
		EXPECT_FALSE(oa::verifyMerkleProof(hashByte(99), proof.getValue(), tree.root));
	}
	EXPECT_TRUE(oa::getMerkleProof(tree, static_cast<oa::U32>(leaves.size())).isError());
}

TEST(Merkle, MalformedDirectionVectorFailsClosed) {
	oa::Vector<oa::Hash> leaves{hashByte(1), hashByte(2)};
	const auto tree = oa::buildMerkleTree(leaves);
	auto proof = oa::getMerkleProof(tree, 0);
	ASSERT_TRUE(proof.isOk());
	proof.getValue().isLeft.clear();
	EXPECT_FALSE(oa::verifyMerkleProof(leaves[0], proof.getValue(), tree.root));
}
