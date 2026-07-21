#pragma once

// Experimental multi-device fragments: heuristic topology/static routing,
// a host-only collective oracle, and external-resource primitives. This umbrella
// is not a supported multi-GPU or cross-machine contract. Prefer individual
// includes while these fragments migrate behind semantic lowering.

#include <Oa/Runtime/Topology.h>
#include <Oa/Runtime/Scheduler.h>
#include <Oa/Runtime/Collective.h>
#include <Oa/Runtime/ExternalMemory.h>
