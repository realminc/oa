#pragma once

#include "../Binding.h"

inline nb::bytes bytes_of(const OaByte* data, OaUsize size) {
    return nb::bytes(reinterpret_cast<const char*>(data), size);
}
