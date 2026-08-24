// OaVma — Vulkan Memory Allocator (hard fork of AMD VMA 3.4.0)
// All Vma*/VMA_* symbols renamed to OaVma*/OA_VMA_*.
// Single compilation TU — includes public API + all implementation files.

#define OA_VMA_STATIC_VULKAN_FUNCTIONS 0
#define OA_VMA_DYNAMIC_VULKAN_FUNCTIONS 0

#include <oa/runtime/oaVk.h>
#include <oa/runtime/oaVma.h>

#include "vma/config.h"
#include "vma/containers.h"
#include "vma/types.h"
#include "vma/metadata.h"
#include "vma/classes.h"
#include "vma/impl.h"
#include "vma/allocator.h"
#include "vma/api.h"
