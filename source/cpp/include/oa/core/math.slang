// Slang math umbrella — `import math;`
// Re-exports shared math modules (Activations, Reductions, Storage, …) for fused shaders.
// Storage uses DTYPE spec constant (legacy). New shaders should import StorageFp32/StorageBf16 directly.
// Importable sources live under source/cpp/lib/oa/Core/Shader/Compute/Math/.

module math;

__exported import activations;
__exported import reductions;
__exported import complex;
__exported import modular;
__exported import keccak;
__exported import storage;
__exported import gemm;
__exported import epilogue;
