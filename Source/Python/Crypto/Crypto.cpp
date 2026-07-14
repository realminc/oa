// OA Python bindings — Crypto registration order.
#include "../Binding.h"

void BindCrypto(nb::module_& m) {
    BindCryptoHash(m);
    BindCryptoSign(m);
    BindCryptoFnHash(m);
}
