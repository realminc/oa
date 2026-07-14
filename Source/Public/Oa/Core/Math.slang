// Slang math umbrella — `import Math;`
// Re-exports shared math modules (Activations, Reductions, Storage, …) for fused shaders.
// Storage uses DTYPE spec constant (legacy). New shaders should import StorageFp32/StorageBf16 directly.
// Importable sources live under Source/Private/Oa/Core/Shader/Compute/Math/.

module Math;

__exported import Activations;
__exported import Reductions;
__exported import Complex;
__exported import Modular;
__exported import Keccak;
__exported import Storage;
__exported import Gemm;
__exported import Epilogue;
