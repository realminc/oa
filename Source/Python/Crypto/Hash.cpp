// OA Python bindings — strict host hashing primitives.
#include "Binding.h"

#include <Oa/Crypto/Hash.h>
#include <Oa/Crypto/Keccak.h>

#include <string>
#include <vector>

void BindCryptoHash(nb::module_& m) {
    // ═════════════════════════════════════════════════════════════════════════
    // OaHash — 32-byte SHAKE-256 digest
    // ═════════════════════════════════════════════════════════════════════════

    nb::class_<OaHash>(m, "OaHash")
        .def(nb::init<>())
        .def("__init__", [](OaHash* self, nb::bytes data) {
			auto result = OaHash::FromBytes(OaSpan<const OaByte>(
				reinterpret_cast<const OaByte*>(data.data()), data.size()));
			throw_if_error(result.GetStatus());
			new (self) OaHash(std::move(result).GetValue());
        }, nb::arg("data"), "Construct from exactly 32 raw bytes.")
		.def_static("FromHex", [](const std::string& hex) {
			auto result = OaHash::FromHex(OaStringView(hex.data(), hex.size()));
			throw_if_error(result.GetStatus());
			return std::move(result).GetValue();
		}, nb::arg("hex"), "Parse exactly 64 hexadecimal characters.")
        .def_static("Zero", &OaHash::Zero)
        .def("ToHex", &OaHash::ToHex)
        .def("ToShortHex", &OaHash::ToShortHex)
        .def("IsZero", &OaHash::IsZero)
        .def("bytes", [](const OaHash& self) { return bytes_of(self.Data(), 32); },
             "The 32 digest bytes.")
        .def("__eq__", [](const OaHash& a, const OaHash& b) { return a == b; })
        .def("__repr__", [](const OaHash& self) {
            return std::string("OaHash(") + self.ToShortHex().c_str() + "...)";
        });

    // ═════════════════════════════════════════════════════════════════════════
    // OaHasher — incremental SHAKE-256
    // ═════════════════════════════════════════════════════════════════════════

    nb::class_<OaHasher>(m, "OaHasher")
        .def(nb::init<>())
        .def("Update", [](OaHasher& self, nb::bytes data) {
			throw_if_error(self.Update(
				reinterpret_cast<const OaByte*>(data.data()), data.size()));
        }, nb::arg("data"), "Absorb more bytes (call repeatedly before Finalize).")
        .def("Finalize", &OaHasher::Finalize, "Squeeze the 32-byte digest.")
        .def("Reset", &OaHasher::Reset);

    // ─── One-shot CPU hashes (single-shot; for batches use the GPU ops below) ──

    m.def("Shake256Bytes", [](nb::bytes data, OaU32 out_len) {
        std::vector<OaByte> out(out_len);
        OaShake256(reinterpret_cast<const OaByte*>(data.data()), data.size(),
                   out.data(), out_len);
        return bytes_of(out.data(), out_len);
    }, nb::arg("data"), nb::arg("out_len") = 32,
       "One-shot SHAKE-256 over bytes → bytes.");

    m.def("Shake128Bytes", [](nb::bytes data, OaU32 out_len) {
        std::vector<OaByte> out(out_len);
        OaShake128(reinterpret_cast<const OaByte*>(data.data()), data.size(),
                   out.data(), out_len);
        return bytes_of(out.data(), out_len);
    }, nb::arg("data"), nb::arg("out_len") = 16,
       "One-shot SHAKE-128 over bytes → bytes.");

    m.def("Kmac256Bytes", [](nb::bytes key, nb::bytes data, nb::bytes custom,
                             OaU32 out_len) {
        std::vector<OaByte> out(out_len);
		throw_if_error(OaKmac256(reinterpret_cast<const OaByte*>(key.data()), key.size(),
				  reinterpret_cast<const OaByte*>(data.data()), data.size(),
				  reinterpret_cast<const OaByte*>(custom.data()), custom.size(),
				  out.data(), out_len));
        return bytes_of(out.data(), out_len);
    }, nb::arg("key"), nb::arg("data"), nb::arg("custom") = nb::bytes("", 0),
       nb::arg("out_len") = 32, "KMAC-256 keyed hash (NIST SP 800-185).");

    m.def("MerkleRootHashes", [](const std::vector<OaHash>& leaves) {
        OaVec<OaHash> v;
        v.reserve(leaves.size());
        for (const auto& h : leaves) { v.push_back(h); }
        return OaMerkleRoot(v);
    }, nb::arg("leaves"), "CPU Merkle root over a list of OaHash leaves.");
}
