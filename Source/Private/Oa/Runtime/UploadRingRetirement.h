#pragma once

#include <Oa/Core/Std.h>
#include <Oa/Runtime/Allocator.h>
#include <Oa/Runtime/Stream.h>

struct OaUploadFrame {
	OaVkStream Stream;
	OaU64 Begin = 0;
	OaU64 End = 0;
	OaU64 Cursor = 0;
};

// Engine-owned lifetime payload for an upload ring abandoned while its Vulkan
// resources may still be referenced. Frame objects stay at their original heap
// addresses so completion tokens that point at their timeline semaphores do not
// become dangling merely because the ring facade was destroyed.
struct OaRetiredUploadRing {
	OaVkBuffer Staging;
	OaVec<OaUniquePtr<OaUploadFrame>> Frames;
};
