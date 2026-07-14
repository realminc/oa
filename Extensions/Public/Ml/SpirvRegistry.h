#pragma once

#include <Oa/Runtime/Spirv.h>

class OaComputeEngine;

// ml's embedded SPIR-V registry (populated at build time via embed_spirv_ml.cmake)
const OaSpvEntry* MlSpvFind(const char* InName);
const OaSpvEntry* MlSpvFindByIndex(OaU32 InIndex);
OaU32 MlSpvCount();

// Register ml's SPIR-V provider with oa's global lookup.
// Call once after OaComputeEngine::Create(), before any model Init().
void MlSpvInit();

// File fallback for liboa / libml kernels when SPIR-V is not embedded (debug presets).
// Call after MlSpvInit(), before model Init(). Safe if embed is on (unused).
void MlAddShaderSearchPaths(OaComputeEngine& InRt);
