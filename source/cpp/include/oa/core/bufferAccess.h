#pragma once

#include <oa/core/types.h>

namespace oa {

enum class BufferAccess : oa::U8 {
	Read,
	Write,
	ReadWrite,
};

} // namespace oa
