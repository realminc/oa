// OA Python bindings — ML-DSA key generation, signing, and verification.
#include "binding.h"

#include <oa/crypto/sign.h>

void bindCryptoSign(nb::module_& m) {
    // ═════════════════════════════════════════════════════════════════════════
    // ML-DSA-65 — CPU keygen/sign/verify (liboqs)
    // ═════════════════════════════════════════════════════════════════════════

	nb::class_<oa::PublicKey>(m, "PublicKey")
		.def("__init__", [](oa::PublicKey* self, nb::bytes data) {
			auto result = oa::deserializePublicKey(oa::Span<const oa::Byte>(
				reinterpret_cast<const oa::Byte*>(data.data()), data.size()));
			throwIfError(result.getStatus());
			new (self) oa::PublicKey(std::move(result).getValue());
		}, nb::arg("data"))
        .def("bytes", [](const oa::PublicKey& self) {
            return bytesOf(self.bytes.data(), OA_SIGN_PUBKEY_SIZE);
        }, "The 1952 public-key bytes.")
        .def("toShortHex", &oa::PublicKey::toShortHex)
        .def("__eq__", [](const oa::PublicKey& a, const oa::PublicKey& b) { return a == b; });

    // Secret key is opaque: never expose its bytes to Python. It is obtained from
    // a keypair and passed straight back into Sign.
    nb::class_<oa::SecretKey>(m, "SecretKey");

	nb::class_<oa::Signature>(m, "Signature")
		.def("__init__", [](oa::Signature* self, nb::bytes data) {
			auto result = oa::deserializeSignature(oa::Span<const oa::Byte>(
				reinterpret_cast<const oa::Byte*>(data.data()), data.size()));
			throwIfError(result.getStatus());
			new (self) oa::Signature(std::move(result).getValue());
        }, nb::arg("data"))
        .def("bytes", [](const oa::Signature& self) {
            return bytesOf(self.bytes.data(), OA_SIGN_SIG_SIZE);
        }, "The 3309 signature bytes.")
        .def("toShortHex", &oa::Signature::toShortHex);

    nb::class_<oa::Keypair>(m, "Keypair")
        .def_prop_ro("pubkey",
            [](oa::Keypair& self) -> oa::PublicKey& { return self.pubkey; },
            nb::rv_policy::reference_internal)
        .def_prop_ro("secret",
            [](oa::Keypair& self) -> oa::SecretKey& { return self.secret; },
            nb::rv_policy::reference_internal);

    m.def("generateKeypair", []() {
        auto r = oa::generateKeypair();
        throwIfError(r.getStatus());
        return new oa::Keypair(std::move(r).getValue());
    }, nb::rv_policy::take_ownership, "Generate a random ML-DSA-65 keypair.");

    m.def("sign", [](nb::bytes data, const oa::SecretKey& secret) {
        auto r = oa::sign(reinterpret_cast<const oa::Byte*>(data.data()), data.size(), secret);
        throwIfError(r.getStatus());
        return new oa::Signature(std::move(r).getValue());
    }, nb::arg("data"), nb::arg("secret"), nb::rv_policy::take_ownership,
       "Sign a message with an ML-DSA-65 secret key.");

    m.def("verify", [](nb::bytes data, const oa::Signature& sig, const oa::PublicKey& pubkey) {
        return static_cast<bool>(oa::verify(
            reinterpret_cast<const oa::Byte*>(data.data()), data.size(), sig, pubkey));
    }, nb::arg("data"), nb::arg("signature"), nb::arg("pubkey"),
       "Verify an ML-DSA-65 signature. Returns False for any tampered input.");
}
