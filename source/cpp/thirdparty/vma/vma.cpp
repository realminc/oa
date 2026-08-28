// Derived Vulkan Memory Allocator 3.4.0 snapshot.
// Single implementation TU; provenance and patch manifest are in UPSTREAM.md.

#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0

#include "vma.h"

#include "detail/config.h"
#include "detail/containers.h"
#include "detail/types.h"
#include "detail/metadata.h"
#include "detail/classes.h"
#include "detail/impl.h"
#include "detail/allocator.h"
#include "detail/api.h"
