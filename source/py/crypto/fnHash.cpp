// OA Python bindings — GPU batch hashing operations.
#include "../binding.h"

#include <oa/crypto/fnHash.h>

void bindCryptoFnHash(nb::module_& m) {
    // Schema-v2 GPU batch operations. Crypto remains an optional module, but
    // its public C++/Python signatures and semantic contracts have one owner.
#include "fnHashOps.gen.inl"
}
