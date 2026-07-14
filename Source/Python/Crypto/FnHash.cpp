// OA Python bindings — GPU batch hashing operations.
#include "../Binding.h"

#include <Oa/Crypto/FnHash.h>

void BindCryptoFnHash(nb::module_& m) {
    // ═════════════════════════════════════════════════════════════════════════
    // OaFnHash — GPU batch ops (auto-context; byte tensors). Flat free funcs.
    // ═════════════════════════════════════════════════════════════════════════

    m.def("Shake256", [](const OaMatrix& messages, OaU32 out_len) {
        return matrix_ptr(OaFnHash::Shake256(messages, out_len));
    }, nb::arg("messages"), nb::arg("out_len") = 32, nb::rv_policy::take_ownership,
       "Batch SHAKE-256 over rows of a [N, MsgLen] u8 matrix → [N, ceil(out/8)*8].");

    m.def("Shake128", [](const OaMatrix& messages, OaU32 out_len) {
        return matrix_ptr(OaFnHash::Shake128(messages, out_len));
    }, nb::arg("messages"), nb::arg("out_len") = 16, nb::rv_policy::take_ownership,
       "Batch SHAKE-128 over rows of a [N, MsgLen] u8 matrix.");

    m.def("KeccakF1600", [](const OaMatrix& states) {
        return matrix_ptr(OaFnHash::KeccakF1600(states));
    }, nb::arg("states"), nb::rv_policy::take_ownership,
       "Batch Keccak-f[1600] permutation over a [N, 200] u8 state matrix.");

    m.def("MerkleRoot", [](const OaMatrix& leaves) {
        return matrix_ptr(OaFnHash::MerkleRoot(leaves));
    }, nb::arg("leaves"), nb::rv_policy::take_ownership,
       "GPU Merkle root of [N, 32] leaves (N a power of two) → [1, 32].");
}
