// TestSign — ML-DSA-65 keygen/sign/verify via liboqs (CPU).
//
// This is the signature security surface. Beyond a happy-path round trip, the
// negative cases are the point: verification MUST reject a tampered message, a
// tampered signature, and a signature checked against the wrong public key.

#include <oa/crypto/sign.h>
#include <oa/crypto/hash.h>
#include <oa/crypto/keccak.h>
#include <oa/crypto/buffer.h>

#include <gtest/gtest.h>

#include <cstring>

namespace {

oa::Keypair makeKeypair() {
	auto kp = oa::generateKeypair();
	EXPECT_TRUE(kp.isOk()) << kp.getStatus().toString().cStr();
	return std::move(kp).getValue();
}

} // namespace

TEST(Sign, KeySizeConstants) {
	EXPECT_EQ(OA_SIGN_PUBKEY_SIZE, 1952);
	EXPECT_EQ(OA_SIGN_SECRET_SIZE, 4032);
	EXPECT_EQ(OA_SIGN_SIG_SIZE, 3309);
}

TEST(Sign, KeygenSignVerifyRoundTrip) {
	oa::Keypair kp = makeKeypair();
	const oa::Byte msg[] = {'O', 'A', ' ', 'M', 'L', '-', 'D', 'S', 'A', '-', '6', '5'};
	auto sig = oa::sign(msg, sizeof(msg), kp.secret);
	ASSERT_TRUE(sig.isOk()) << sig.getStatus().toString().cStr();
	EXPECT_TRUE(oa::verify(msg, sizeof(msg), sig.getValue(), kp.pubkey));
}

TEST(Sign, TamperedMessageRejected) {
	oa::Keypair kp = makeKeypair();
	oa::Byte msg[32];
	for (oa::I32 i = 0; i < 32; ++i) { msg[i] = static_cast<oa::Byte>(i); }
	auto sig = oa::sign(msg, sizeof(msg), kp.secret);
	ASSERT_TRUE(sig.isOk());
	ASSERT_TRUE(oa::verify(msg, sizeof(msg), sig.getValue(), kp.pubkey));

	msg[5] = static_cast<oa::Byte>(msg[5] ^ 0x01);
	EXPECT_FALSE(oa::verify(msg, sizeof(msg), sig.getValue(), kp.pubkey));
}

TEST(Sign, TamperedSignatureRejected) {
	oa::Keypair kp = makeKeypair();
	const oa::Byte msg[] = {'t', 'a', 'm', 'p', 'e', 'r'};
	auto sig = oa::sign(msg, sizeof(msg), kp.secret);
	ASSERT_TRUE(sig.isOk());

	oa::Signature bad = sig.getValue();
	bad.bytes[0] = static_cast<oa::Byte>(bad.bytes[0] ^ 0x01);
	EXPECT_FALSE(oa::verify(msg, sizeof(msg), bad, kp.pubkey));
}

TEST(Sign, WrongKeyRejected) {
	oa::Keypair a = makeKeypair();
	oa::Keypair b = makeKeypair();
	const oa::Byte msg[] = {'w', 'r', 'o', 'n', 'g', 'k', 'e', 'y'};
	auto sig = oa::sign(msg, sizeof(msg), a.secret);
	ASSERT_TRUE(sig.isOk());
	EXPECT_TRUE(oa::verify(msg, sizeof(msg), sig.getValue(), a.pubkey));
	EXPECT_FALSE(oa::verify(msg, sizeof(msg), sig.getValue(), b.pubkey));
}

TEST(Sign, SignHashOverload) {
	oa::Keypair kp = makeKeypair();
	const oa::Byte data[] = {'h', 'a', 's', 'h', 'm', 'e'};
	oa::Hash h;
	oa::shake256(data, sizeof(data), h.bytes.data(), 32);
	auto sig = oa::sign(h, kp.secret);
	ASSERT_TRUE(sig.isOk());
	EXPECT_TRUE(oa::verify(h, sig.getValue(), kp.pubkey));

	// A different hash must not verify against this signature.
	oa::Hash other;
	const oa::Byte data2[] = {'o', 't', 'h', 'e', 'r'};
	oa::shake256(data2, sizeof(data2), other.bytes.data(), 32);
	EXPECT_FALSE(oa::verify(other, sig.getValue(), kp.pubkey));
}

TEST(Sign, SerializeDeserializeRoundTrip) {
	oa::Keypair kp = makeKeypair();
	const oa::Byte msg[] = {'s', 'e', 'r', 'i', 'a', 'l'};
	auto sig = oa::sign(msg, sizeof(msg), kp.secret);
	ASSERT_TRUE(sig.isOk());

	auto pkBuf = oa::serializePublicKey(kp.pubkey);
	auto sigBuf = oa::serializeSignature(sig.getValue());
	auto pkResult = oa::deserializePublicKey(pkBuf);
	auto sigResult = oa::deserializeSignature(sigBuf);
	ASSERT_TRUE(pkResult.isOk());
	ASSERT_TRUE(sigResult.isOk());
	oa::PublicKey pk2 = std::move(pkResult).getValue();
	oa::Signature sig2 = std::move(sigResult).getValue();

	EXPECT_TRUE(pk2 == kp.pubkey);
	EXPECT_TRUE(oa::verify(msg, sizeof(msg), sig2, pk2));
}

TEST(Sign, DeserializeRejectsWrongLengths) {
	oa::Byte shortPk[OA_SIGN_PUBKEY_SIZE - 1]{};
	oa::Byte shortSig[OA_SIGN_SIG_SIZE - 1]{};
	EXPECT_TRUE(oa::deserializePublicKey(shortPk).isError());
	EXPECT_TRUE(oa::deserializeSignature(shortSig).isError());
	EXPECT_TRUE(oa::deserializePublicKey(
		oa::Span<const oa::Byte>(nullptr, OA_SIGN_PUBKEY_SIZE)).isError());
	EXPECT_TRUE(oa::deserializeSignature(
		oa::Span<const oa::Byte>(nullptr, OA_SIGN_SIG_SIZE)).isError());
}

TEST(Sign, EmptyMessageRoundTrip) {
	oa::Keypair kp = makeKeypair();
	auto sig = oa::sign(nullptr, 0, kp.secret);
	ASSERT_TRUE(sig.isOk());
	EXPECT_TRUE(oa::verify(nullptr, 0, sig.getValue(), kp.pubkey));
}

TEST(Sign, NullNonEmptyMessageRejected) {
	oa::Keypair kp = makeKeypair();
	EXPECT_TRUE(oa::sign(nullptr, 1, kp.secret).isError());
	oa::Signature empty;
	EXPECT_FALSE(oa::verify(nullptr, 1, empty, kp.pubkey));
}

TEST(Sign, DistinctKeypairsDiffer) {
	oa::Keypair a = makeKeypair();
	oa::Keypair b = makeKeypair();
	EXPECT_FALSE(a.pubkey == b.pubkey);
}

TEST(SecureBuffer, ZeroOnDestroy) {
	alignas(64) oa::U8 backing[256];
	oa::memset(backing, 0xAA, sizeof(backing));

	{
		oa::SecureBuffer secure(backing, sizeof(backing));
		EXPECT_TRUE(secure.isValid());
	}

	for (oa::Byte byte : backing) {
		EXPECT_EQ(byte, 0u);
	}
}

TEST(SecureBuffer, MoveSemantics) {
	alignas(64) oa::U8 backing[128];
	oa::memset(backing, 0xBB, sizeof(backing));

	oa::SecureBuffer source(backing, sizeof(backing));
	ASSERT_TRUE(source.isValid());

	oa::SecureBuffer destination = std::move(source);
	EXPECT_FALSE(source.isValid());
	EXPECT_TRUE(destination.isValid());
	EXPECT_EQ(destination.data(), backing);
	EXPECT_EQ(destination.sizeBytes(), sizeof(backing));

	destination.reset();
	EXPECT_FALSE(destination.isValid());
	for (oa::Byte byte : backing) {
		EXPECT_EQ(byte, 0u);
	}
}

TEST(SecureBuffer, DefaultIsInvalid) {
	oa::SecureBuffer secure;
	EXPECT_FALSE(secure.isValid());
}
