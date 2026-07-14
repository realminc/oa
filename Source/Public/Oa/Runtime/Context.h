#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/Status.h>
#include <Oa/Core/BufferAccess.h>
#include <Oa/Runtime/Allocator.h>
#include <Oa/Runtime/Sync.h>

#include <initializer_list>

// OaContext — Unified execution context for all Oa compute operations.
// Declarations live in Source/Private/Oa/Runtime/Context/ and are included
// through the private umbrella below.
#include "../../../Private/Oa/Runtime/Context/ContextPrivate.h"
