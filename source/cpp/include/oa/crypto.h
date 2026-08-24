// OA Crypto — Umbrella header
// One primitive (Keccak-f[1600]): SHAKE-256/128 hashing, incremental hasher,
// Merkle trees, KMAC-256, and typed ML-DSA-65 signing via liboqs. Plus the
// verified oa::FnHash Vulkan batch surface and secure buffer views. Experimental
// vulkan ML-DSA kernels are intentionally not part of this umbrella.

#pragma once

#include <oa/crypto/fnHash.h>
#include <oa/crypto/hash.h>
#include <oa/crypto/keccak.h>
#include <oa/crypto/sign.h>
#include <oa/crypto/buffer.h>
