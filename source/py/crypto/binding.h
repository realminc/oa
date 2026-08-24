#pragma once

#include "../binding.h"

inline nb::bytes bytesOf(const oa::Byte* data, oa::Usize size) {
    return nb::bytes(reinterpret_cast<const char*>(data), size);
}
