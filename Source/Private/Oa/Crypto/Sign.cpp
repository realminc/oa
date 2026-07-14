// ML-DSA-65 CPU implementation via liboqs.

#include <Oa/Crypto/Sign.h>
#include <oqs/oqs.h>

OaResult<OaKeypair> OaGenerateKeypair() {
	OQS_SIG* Sig = OQS_SIG_new(OQS_SIG_alg_ml_dsa_65);
	if (!Sig) {
		return OaStatus::Error(OaStatusCode::Internal, "Failed to create ML-DSA-65 context");
	}
	if (Sig->length_public_key != OA_SIGN_PUBKEY_SIZE ||
		Sig->length_secret_key != OA_SIGN_SECRET_SIZE ||
		Sig->length_signature != OA_SIGN_SIG_SIZE) {
		OQS_SIG_free(Sig);
		return OaStatus::Error(OaStatusCode::FailedPrecondition,
			"liboqs ML-DSA-65 sizes do not match the OA ABI");
	}

	OaKeypair Kp;
	OQS_STATUS Status = OQS_SIG_keypair(Sig, Kp.Pubkey.Bytes.data(), Kp.Secret.Bytes.data());
	OQS_SIG_free(Sig);

	if (Status != OQS_SUCCESS) {
		return OaStatus::Error(OaStatusCode::Internal, "ML-DSA-65 keygen failed");
	}

	return Kp;
}

// NOTE: deterministic keygen from a 32-byte seed is intentionally NOT provided.
// liboqs 0.15 exposes only the random OQS_SIG_ml_dsa_65_keypair(); ML-DSA's
// seed-derived (ξ) keygen has no public liboqs entry point, and faking it by
// hijacking the global OQS RNG is not thread-safe. Reintroduce a real
// OaGenerateKeypairFromSeed once liboqs exposes ml_dsa_65_keypair_derand.

OaResult<OaSignature> OaSign(
		const OaByte* InMessage,
		OaUsize InMessageLen,
		const OaSecretKey& InSecret
) {
	if (InMessage == nullptr && InMessageLen != 0) {
		return OaStatus::InvalidArgument("ML-DSA-65 message is null with a non-zero length");
	}
	OQS_SIG* Sig = OQS_SIG_new(OQS_SIG_alg_ml_dsa_65);
	if (!Sig) {
		return OaStatus::Error(OaStatusCode::Internal, "Failed to create ML-DSA-65 context");
	}

	OaSignature OutSig;
	size_t SigLen = OA_SIGN_SIG_SIZE;
	OQS_STATUS Status = OQS_SIG_sign(
		Sig,
		OutSig.Bytes.data(),
		&SigLen,
		InMessage,
		InMessageLen,
		InSecret.Bytes.data()
	);
	OQS_SIG_free(Sig);

	if (Status != OQS_SUCCESS || SigLen != OA_SIGN_SIG_SIZE) {
		return OaStatus::Error(OaStatusCode::Internal, "ML-DSA-65 sign failed");
	}

	return OutSig;
}

OaBool OaVerify(
		const OaByte* InMessage,
		OaUsize InMessageLen,
		const OaSignature& InSignature,
		const OaPublicKey& InPubkey
) {
	if (InMessage == nullptr && InMessageLen != 0) {
		return false;
	}
	OQS_SIG* Sig = OQS_SIG_new(OQS_SIG_alg_ml_dsa_65);
	if (!Sig) {
		return false;
	}

	OQS_STATUS Status = OQS_SIG_verify(
		Sig,
		InMessage,
		InMessageLen,
		InSignature.Bytes.data(),
		OA_SIGN_SIG_SIZE,
		InPubkey.Bytes.data()
	);
	OQS_SIG_free(Sig);

	return Status == OQS_SUCCESS;
}
