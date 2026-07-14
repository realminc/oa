// OA Python bindings — ML-DSA key generation, signing, and verification.
#include "Binding.h"

#include <Oa/Crypto/Sign.h>

void BindCryptoSign(nb::module_& m) {
    // ═════════════════════════════════════════════════════════════════════════
    // ML-DSA-65 — CPU keygen/sign/verify (liboqs)
    // ═════════════════════════════════════════════════════════════════════════

	nb::class_<OaPublicKey>(m, "OaPublicKey")
		.def("__init__", [](OaPublicKey* self, nb::bytes data) {
			auto result = OaDeserializePublicKey(OaSpan<const OaByte>(
				reinterpret_cast<const OaByte*>(data.data()), data.size()));
			throw_if_error(result.GetStatus());
			new (self) OaPublicKey(std::move(result).GetValue());
		}, nb::arg("data"))
        .def("bytes", [](const OaPublicKey& self) {
            return bytes_of(self.Bytes.data(), OA_SIGN_PUBKEY_SIZE);
        }, "The 1952 public-key bytes.")
        .def("ToShortHex", &OaPublicKey::ToShortHex)
        .def("__eq__", [](const OaPublicKey& a, const OaPublicKey& b) { return a == b; });

    // Secret key is opaque: never expose its bytes to Python. It is obtained from
    // a keypair and passed straight back into Sign.
    nb::class_<OaSecretKey>(m, "OaSecretKey");

	nb::class_<OaSignature>(m, "OaSignature")
		.def("__init__", [](OaSignature* self, nb::bytes data) {
			auto result = OaDeserializeSignature(OaSpan<const OaByte>(
				reinterpret_cast<const OaByte*>(data.data()), data.size()));
			throw_if_error(result.GetStatus());
			new (self) OaSignature(std::move(result).GetValue());
        }, nb::arg("data"))
        .def("bytes", [](const OaSignature& self) {
            return bytes_of(self.Bytes.data(), OA_SIGN_SIG_SIZE);
        }, "The 3309 signature bytes.")
        .def("ToShortHex", &OaSignature::ToShortHex);

    nb::class_<OaKeypair>(m, "OaKeypair")
        .def_prop_ro("Pubkey",
            [](OaKeypair& self) -> OaPublicKey& { return self.Pubkey; },
            nb::rv_policy::reference_internal)
        .def_prop_ro("Secret",
            [](OaKeypair& self) -> OaSecretKey& { return self.Secret; },
            nb::rv_policy::reference_internal);

    m.def("GenerateKeypair", []() {
        auto r = OaGenerateKeypair();
        throw_if_error(r.GetStatus());
        return new OaKeypair(std::move(r).GetValue());
    }, nb::rv_policy::take_ownership, "Generate a random ML-DSA-65 keypair.");

    m.def("Sign", [](nb::bytes data, const OaSecretKey& secret) {
        auto r = OaSign(reinterpret_cast<const OaByte*>(data.data()), data.size(), secret);
        throw_if_error(r.GetStatus());
        return new OaSignature(std::move(r).GetValue());
    }, nb::arg("data"), nb::arg("secret"), nb::rv_policy::take_ownership,
       "Sign a message with an ML-DSA-65 secret key.");

    m.def("Verify", [](nb::bytes data, const OaSignature& sig, const OaPublicKey& pubkey) {
        return static_cast<bool>(OaVerify(
            reinterpret_cast<const OaByte*>(data.data()), data.size(), sig, pubkey));
    }, nb::arg("data"), nb::arg("signature"), nb::arg("pubkey"),
       "Verify an ML-DSA-65 signature. Returns False for any tampered input.");
}
