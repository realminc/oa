// UI-composition configuration for the canonical oa::Renderer session.

#pragma once

#include <oa/core/types.h>
#include <oa/ui/style.h>

namespace oa {

class UiRenderConfig {
public:
	oa::U32 width_ = 1280U;
	oa::U32 height_ = 720U;
	oa::U32 targetSlotCount_ = 3U;
	UiStyle style_ = UiStyle::editorDark();
};

}  // namespace oa
