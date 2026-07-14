#include <Oa/Runtime/Engine.h>

// OaEngine: host / policy base. Special members are defined inline in Engine.h today.
// Future host-side state (e.g. OaDgGraph for CPU DG eval per dg.mdc) belongs here, not on
// OaComputeEngine, so render-only and compute-only engines can share the same host layer.
