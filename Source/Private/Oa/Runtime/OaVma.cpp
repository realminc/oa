// OaVma — Vulkan Memory Allocator (hard fork of AMD VMA 3.4.0)
// All Vma*/VMA_* symbols renamed to OaVma*/OA_VMA_*.
// Single compilation TU — includes public API + all implementation files.

#define OA_VMA_STATIC_VULKAN_FUNCTIONS 0
#define OA_VMA_DYNAMIC_VULKAN_FUNCTIONS 0

#include <Oa/Runtime/OaVk.h>
#include <Oa/Runtime/OaVma.h>

#include "Vma/Config.h"
#include "Vma/Containers.h"
#include "Vma/Types.h"
#include "Vma/Metadata.h"
#include "Vma/Classes.h"
#include "Vma/Impl.h"
#include "Vma/Allocator.h"
#include "Vma/Api.h"
