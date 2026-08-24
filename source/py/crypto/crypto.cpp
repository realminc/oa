// OA Python bindings — Crypto registration order.
#include "../binding.h"

void bindCrypto(nb::module_& m, nb::module_& inFnHash) {
    bindCryptoHash(m);
    bindCryptoSign(m);
    bindCryptoFnHash(inFnHash);
}
