#pragma once

// OaStd — umbrella for header-only templates under Source/Public/Oa/Core/Std/ (VMA-style).
// Bridge phase: std semantics, OA naming — Oa* types, PascalCase methods on OaStd*
// (OaVec alone uses snake_case for std::vector parity). See docs/OaStd.md.
// Mix: native OaVec, OaStdString, hash maps, shared_ptr/weak_ptr, etc.; some modules still compose std internally.
// Each `Source/Public/Oa/Core/Std/*.h` file summarizes behavior, asserts vs throws, and `Std*()` boundaries at the top.

#include <Oa/Core/Std/Chrono.h>
#include <Oa/Core/Std/Allocator.h>
#include <Oa/Core/Std/StringView.h>
#include <Oa/Core/Std/Array.h>
#include <Oa/Core/Std/Optional.h>
#include <Oa/Core/Std/UniquePtr.h>
#include <Oa/Core/Std/SharedPtr.h>
#include <Oa/Core/Std/Variant.h>
#include <Oa/Core/Std/Function.h>
#include <Oa/Core/Std/HashMap.h>
#include <Oa/Core/Std/Span.h>
#include <Oa/Core/Std/String.h>
#include <Oa/Core/Std/Format.h>
#include <Oa/Core/Std/Path.h>
#include <Oa/Core/Std/Filesystem.h>
#include <Oa/Core/Std/Algo.h>
#include <Oa/Core/Std/Vec.h>
#include <Oa/Core/Std/Random.h>
#include <Oa/Core/Std/Sync.h>
#include <Oa/Core/Std/ScalarMath.h>
#include <Oa/Core/Std/CString.h>
#include <Oa/Core/Std/Pair.h>
#include <Oa/Core/Std/Limits.h>
