// ML-DSA-65 CPU implementation via liboqs.

#include <oa/crypto/sign.h>
#include <oqs/oqs.h>

namespace oa {

Result<Keypair> generateKeypair() {
	OQS_SIG* sig = OQS_SIG_new(OQS_SIG_alg_ml_dsa_65);
	if (!sig) {
		return oa::Status::error(oa::StatusCode::Internal, "Failed to create ML-DSA-65 context");
	}
	if (sig->length_public_key != OA_SIGN_PUBKEY_SIZE ||
		sig->length_secret_key != OA_SIGN_SECRET_SIZE ||
		sig->length_signature != OA_SIGN_SIG_SIZE) {
		OQS_SIG_free(sig);
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"liboqs ML-DSA-65 sizes do not match the OA ABI");
	}

	Keypair keypair;
	OQS_STATUS status = OQS_SIG_keypair(
		sig, keypair.pubkey.bytes.data(), keypair.secret.bytes.data());
	OQS_SIG_free(sig);

	if (status != OQS_SUCCESS) {
		return oa::Status::error(oa::StatusCode::Internal, "ML-DSA-65 keygen failed");
	}

	return keypair;
}

// NOTE: deterministic keygen from a 32-byte seed is intentionally NOT provided.
// liboqs 0.15 exposes only the random oQS_SIG_ml_dsa_65_keypair(); ML-DSA's
// seed-derived (ξ) keygen has no public liboqs entry point, and faking it by
// hijacking the global OQS RNG is not thread-safe. Reintroduce a real
// A deterministic generateKeypairFromSeed can be added once liboqs exposes
// ml_dsa_65_keypair_derand.

Result<Signature> sign(
		const oa::Byte* inMessage,
		oa::Usize inMessageLen,
		const SecretKey& inSecret
) {
	if (inMessage == nullptr && inMessageLen != 0) {
		return oa::Status::invalidArgument("ML-DSA-65 message is null with a non-zero length");
	}
	OQS_SIG* sig = OQS_SIG_new(OQS_SIG_alg_ml_dsa_65);
	if (!sig) {
		return oa::Status::error(oa::StatusCode::Internal, "Failed to create ML-DSA-65 context");
	}

	Signature outSig;
	size_t sigLen = OA_SIGN_SIG_SIZE;
	OQS_STATUS status = OQS_SIG_sign(
		sig,
		outSig.bytes.data(),
		&sigLen,
		inMessage,
		inMessageLen,
		inSecret.bytes.data()
	);
	OQS_SIG_free(sig);

	if (status != OQS_SUCCESS || sigLen != OA_SIGN_SIG_SIZE) {
		return oa::Status::error(oa::StatusCode::Internal, "ML-DSA-65 sign failed");
	}

	return outSig;
}

Bool verify(
		const oa::Byte* inMessage,
		oa::Usize inMessageLen,
		const Signature& inSignature,
		const PublicKey& inPubkey
) {
	if (inMessage == nullptr && inMessageLen != 0) {
		return false;
	}
	OQS_SIG* sig = OQS_SIG_new(OQS_SIG_alg_ml_dsa_65);
	if (!sig) {
		return false;
	}

	OQS_STATUS status = OQS_SIG_verify(
		sig,
		inMessage,
		inMessageLen,
		inSignature.bytes.data(),
		OA_SIGN_SIG_SIZE,
		inPubkey.bytes.data()
	);
	OQS_SIG_free(sig);

	return status == OQS_SUCCESS;
}

} // namespace oa
