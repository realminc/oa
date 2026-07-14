#pragma once

// Internal runtime-global access — not for OaFn bodies.
//
// OaFn implementations (tensor / loss / grad / optim) must record through
// OaContext::GetDefault(). Use this header only from:
//   - OaComputeEngine init/teardown (engine wires itself in as the global)
//   - The allocator inside OaFnMatrix (Empty / EmptyOn / Zeros / Rand / …)
//   - The autogen oafnautogen.py test scaffolding
//   - Test fixtures that bring up a runtime without a full engine init
//
// Anything else should call OaContext::GetDefault().GetRuntime() instead.

class OaComputeEngine;

namespace OaRuntimeGlobal {

void SetRuntime(OaComputeEngine* InRuntime);
[[nodiscard]] OaComputeEngine* GetRuntime();

} // namespace OaRuntimeGlobal
