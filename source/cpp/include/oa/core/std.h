#pragma once

// OA standard-library umbrella for header-only templates under
// source/cpp/include/oa/core/std/. Types use PascalCase and methods use
// camelCase.
// The implementation mixes native oa::Vector, oa::String, maps and smart pointers;
// some modules still compose std internally at explicit boundaries.

#include <oa/core/std/assert.h>
#include <oa/core/std/chrono.h>
#include <oa/core/std/allocator.h>
#include <oa/core/std/memory.h>
#include <oa/core/std/lifetime.h>
#include <oa/core/std/stringView.h>
#include <oa/core/std/array.h>
#include <oa/core/std/atomic.h>
#include <oa/core/std/optional.h>
#include <oa/core/std/uniquePtr.h>
#include <oa/core/std/sharedPtr.h>
#include <oa/core/std/variant.h>
#include <oa/core/std/function.h>
#include <oa/core/std/keyHash.h>
#include <oa/core/std/hashMap.h>
#include <oa/core/std/span.h>
#include <oa/core/std/string.h>
#include <oa/core/std/format.h>
#include <oa/core/std/path.h>
#include <oa/core/std/algo.h>
#include <oa/core/std/vector.h>
#include <oa/core/std/random.h>
#include <oa/core/std/sync.h>
#include <oa/core/std/scalarMath.h>
#include <oa/core/std/cString.h>
#include <oa/core/std/pair.h>
#include <oa/core/std/limits.h>
