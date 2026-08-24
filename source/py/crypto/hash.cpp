// OA Python bindings — strict host hashing primitives.
#include "binding.h"

#include <oa/crypto/hash.h>
#include <oa/crypto/keccak.h>

#include <string>
#include <vector>

void bindCryptoHash(nb::module_& m) {
    // ═════════════════════════════════════════════════════════════════════════
    // oa::Hash — 32-byte SHAKE-256 digest
    // ═════════════════════════════════════════════════════════════════════════

    nb::class_<oa::Hash>(m, "Hash")
        .def(nb::init<>())
        .def("__init__", [](oa::Hash* self, nb::bytes data) {
			auto result = oa::Hash::fromBytes(oa::Span<const oa::Byte>(
				reinterpret_cast<const oa::Byte*>(data.data()), data.size()));
			throwIfError(result.getStatus());
			new (self) oa::Hash(std::move(result).getValue());
        }, nb::arg("data"), "Construct from exactly 32 raw bytes.")
		.def_static("fromHex", [](const std::string& hex) {
			auto result = oa::Hash::fromHex(oa::StringView(hex.data(), hex.size()));
			throwIfError(result.getStatus());
			return std::move(result).getValue();
		}, nb::arg("hex"), "Parse exactly 64 hexadecimal characters.")
        .def_static("zero", &oa::Hash::zero)
        .def("toHex", &oa::Hash::toHex)
        .def("toShortHex", &oa::Hash::toShortHex)
        .def("isZero", &oa::Hash::isZero)
        .def("bytes", [](const oa::Hash& self) { return bytesOf(self.data(), 32); },
             "The 32 digest bytes.")
        .def("__eq__", [](const oa::Hash& a, const oa::Hash& b) { return a == b; })
        .def("__repr__", [](const oa::Hash& self) {
            return std::string("oa.Hash(") + self.toShortHex().cStr() + "...)";
        });

    // ═════════════════════════════════════════════════════════════════════════
    // oa::Hasher — incremental SHAKE-256
    // ═════════════════════════════════════════════════════════════════════════

    nb::class_<oa::Hasher>(m, "Hasher")
        .def(nb::init<>())
        .def("update", [](oa::Hasher& self, nb::bytes data) {
			throwIfError(self.update(
				reinterpret_cast<const oa::Byte*>(data.data()), data.size()));
        }, nb::arg("data"), "Absorb more bytes (call repeatedly before Finalize).")
        .def("finalize", &oa::Hasher::finalize, "Squeeze the 32-byte digest.")
        .def("reset", &oa::Hasher::reset);

    // ─── One-shot CPU hashes (single-shot; for batches use the GPU ops below) ──

    m.def("shake256", [](nb::bytes data, oa::U32 outLen) {
        std::vector<oa::Byte> out(outLen);
        oa::shake256(reinterpret_cast<const oa::Byte*>(data.data()), data.size(),
                   out.data(), outLen);
        return bytesOf(out.data(), outLen);
    }, nb::arg("data"), nb::arg("outLen") = 32,
       "One-shot SHAKE-256 over bytes → bytes.");

    m.def("shake128", [](nb::bytes data, oa::U32 outLen) {
        std::vector<oa::Byte> out(outLen);
        oa::shake128(reinterpret_cast<const oa::Byte*>(data.data()), data.size(),
                   out.data(), outLen);
        return bytesOf(out.data(), outLen);
    }, nb::arg("data"), nb::arg("outLen") = 16,
       "One-shot SHAKE-128 over bytes → bytes.");

    m.def("kmac256", [](nb::bytes key, nb::bytes data, nb::bytes custom,
                             oa::U32 outLen) {
        std::vector<oa::Byte> out(outLen);
		throwIfError(oa::kmac256(reinterpret_cast<const oa::Byte*>(key.data()), key.size(),
				  reinterpret_cast<const oa::Byte*>(data.data()), data.size(),
				  reinterpret_cast<const oa::Byte*>(custom.data()), custom.size(),
				  out.data(), outLen));
        return bytesOf(out.data(), outLen);
    }, nb::arg("key"), nb::arg("data"), nb::arg("custom") = nb::bytes("", 0),
       nb::arg("outLen") = 32, "KMAC-256 keyed hash (NIST SP 800-185).");

    m.def("merkleRoot", [](const std::vector<oa::Hash>& leaves) {
        oa::Vec<oa::Hash> v;
        v.reserve(leaves.size());
        for (const auto& h : leaves) { v.pushBack(h); }
        return oa::merkleRoot(v);
    }, nb::arg("leaves"), "CPU Merkle root over a list of oa::Hash leaves.");
}
