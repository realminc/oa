// TestSign — ML-DSA-65 keygen/sign/verify via liboqs (CPU).
//
// This is the signature security surface. Beyond a happy-path round trip, the
// negative cases are the point: verification MUST reject a tampered message, a
// tampered signature, and a signature checked against the wrong public key.

#include <Oa/Crypto/Sign.h>
#include <Oa/Crypto/Hash.h>
#include <Oa/Crypto/Keccak.h>

#include <gtest/gtest.h>

#include <cstring>

namespace {

OaKeypair MakeKeypair() {
	auto kp = OaGenerateKeypair();
	EXPECT_TRUE(kp.IsOk()) << kp.GetStatus().ToString().c_str();
	return std::move(kp).GetValue();
}

} // namespace

TEST(Sign, KeySizeConstants) {
	EXPECT_EQ(OA_SIGN_PUBKEY_SIZE, 1952);
	EXPECT_EQ(OA_SIGN_SECRET_SIZE, 4032);
	EXPECT_EQ(OA_SIGN_SIG_SIZE, 3309);
}

TEST(Sign, KeygenSignVerifyRoundTrip) {
	OaKeypair kp = MakeKeypair();
	const OaByte msg[] = {'O', 'A', ' ', 'M', 'L', '-', 'D', 'S', 'A', '-', '6', '5'};
	auto sig = OaSign(msg, sizeof(msg), kp.Secret);
	ASSERT_TRUE(sig.IsOk()) << sig.GetStatus().ToString().c_str();
	EXPECT_TRUE(OaVerify(msg, sizeof(msg), sig.GetValue(), kp.Pubkey));
}

TEST(Sign, TamperedMessageRejected) {
	OaKeypair kp = MakeKeypair();
	OaByte msg[32];
	for (OaI32 i = 0; i < 32; ++i) { msg[i] = static_cast<OaByte>(i); }
	auto sig = OaSign(msg, sizeof(msg), kp.Secret);
	ASSERT_TRUE(sig.IsOk());
	ASSERT_TRUE(OaVerify(msg, sizeof(msg), sig.GetValue(), kp.Pubkey));

	msg[5] = static_cast<OaByte>(msg[5] ^ 0x01);
	EXPECT_FALSE(OaVerify(msg, sizeof(msg), sig.GetValue(), kp.Pubkey));
}

TEST(Sign, TamperedSignatureRejected) {
	OaKeypair kp = MakeKeypair();
	const OaByte msg[] = {'t', 'a', 'm', 'p', 'e', 'r'};
	auto sig = OaSign(msg, sizeof(msg), kp.Secret);
	ASSERT_TRUE(sig.IsOk());

	OaSignature bad = sig.GetValue();
	bad.Bytes[0] = static_cast<OaByte>(bad.Bytes[0] ^ 0x01);
	EXPECT_FALSE(OaVerify(msg, sizeof(msg), bad, kp.Pubkey));
}

TEST(Sign, WrongKeyRejected) {
	OaKeypair a = MakeKeypair();
	OaKeypair b = MakeKeypair();
	const OaByte msg[] = {'w', 'r', 'o', 'n', 'g', 'k', 'e', 'y'};
	auto sig = OaSign(msg, sizeof(msg), a.Secret);
	ASSERT_TRUE(sig.IsOk());
	EXPECT_TRUE(OaVerify(msg, sizeof(msg), sig.GetValue(), a.Pubkey));
	EXPECT_FALSE(OaVerify(msg, sizeof(msg), sig.GetValue(), b.Pubkey));
}

TEST(Sign, SignHashOverload) {
	OaKeypair kp = MakeKeypair();
	const OaByte data[] = {'h', 'a', 's', 'h', 'm', 'e'};
	OaHash h;
	OaShake256(data, sizeof(data), h.Bytes.data(), 32);
	auto sig = OaSign(h, kp.Secret);
	ASSERT_TRUE(sig.IsOk());
	EXPECT_TRUE(OaVerify(h, sig.GetValue(), kp.Pubkey));

	// A different hash must not verify against this signature.
	OaHash other;
	const OaByte data2[] = {'o', 't', 'h', 'e', 'r'};
	OaShake256(data2, sizeof(data2), other.Bytes.data(), 32);
	EXPECT_FALSE(OaVerify(other, sig.GetValue(), kp.Pubkey));
}

TEST(Sign, SerializeDeserializeRoundTrip) {
	OaKeypair kp = MakeKeypair();
	const OaByte msg[] = {'s', 'e', 'r', 'i', 'a', 'l'};
	auto sig = OaSign(msg, sizeof(msg), kp.Secret);
	ASSERT_TRUE(sig.IsOk());

	auto pkBuf = OaSerializePublicKey(kp.Pubkey);
	auto sigBuf = OaSerializeSignature(sig.GetValue());
	auto pkResult = OaDeserializePublicKey(pkBuf);
	auto sigResult = OaDeserializeSignature(sigBuf);
	ASSERT_TRUE(pkResult.IsOk());
	ASSERT_TRUE(sigResult.IsOk());
	OaPublicKey pk2 = std::move(pkResult).GetValue();
	OaSignature sig2 = std::move(sigResult).GetValue();

	EXPECT_TRUE(pk2 == kp.Pubkey);
	EXPECT_TRUE(OaVerify(msg, sizeof(msg), sig2, pk2));
}

TEST(Sign, DeserializeRejectsWrongLengths) {
	OaByte shortPk[OA_SIGN_PUBKEY_SIZE - 1]{};
	OaByte shortSig[OA_SIGN_SIG_SIZE - 1]{};
	EXPECT_TRUE(OaDeserializePublicKey(shortPk).IsError());
	EXPECT_TRUE(OaDeserializeSignature(shortSig).IsError());
	EXPECT_TRUE(OaDeserializePublicKey(
		OaSpan<const OaByte>(nullptr, OA_SIGN_PUBKEY_SIZE)).IsError());
	EXPECT_TRUE(OaDeserializeSignature(
		OaSpan<const OaByte>(nullptr, OA_SIGN_SIG_SIZE)).IsError());
}

TEST(Sign, EmptyMessageRoundTrip) {
	OaKeypair kp = MakeKeypair();
	auto sig = OaSign(nullptr, 0, kp.Secret);
	ASSERT_TRUE(sig.IsOk());
	EXPECT_TRUE(OaVerify(nullptr, 0, sig.GetValue(), kp.Pubkey));
}

TEST(Sign, NullNonEmptyMessageRejected) {
	OaKeypair kp = MakeKeypair();
	EXPECT_TRUE(OaSign(nullptr, 1, kp.Secret).IsError());
	OaSignature empty;
	EXPECT_FALSE(OaVerify(nullptr, 1, empty, kp.Pubkey));
}

TEST(Sign, DistinctKeypairsDiffer) {
	OaKeypair a = MakeKeypair();
	OaKeypair b = MakeKeypair();
	EXPECT_FALSE(a.Pubkey == b.Pubkey);
}
