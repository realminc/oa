#pragma once

#include <oa/core/std.h>
#include <oa/runtime/allocator.h>
#include <oa/runtime/stream.h>

namespace oa {

struct UploadFrame {
	oavk::Stream stream;
	oa::U64 begin = 0;
	oa::U64 end = 0;
	oa::U64 cursor = 0;
};

// Engine-owned lifetime payload for an upload ring abandoned while its vulkan
// resources may still be referenced. Frame objects stay at their original heap
// addresses so completion events that point at their timeline semaphores do not
// become dangling merely because the ring facade was destroyed.
struct RetiredUploadRing {
	oavk::Buffer staging;
	oa::Vec<oa::UniquePtr<oa::UploadFrame>> frames;
};

} // namespace oa
