// Mobile specialization for the NLP-suite Mamba-3 shape.
//
// Turnip cannot compile either of the existing choices reliably:
//   - Mamba3SisoBwdShort reserves a 32 KiB workgroup state history;
//   - the generic kernel carries MAXN=128 private state arrays.
//
// Keep the exact verified recurrence, bound private arrays to the real mobile
// shape, and use the existing global ChunkBuf path for recomputed state.
#define OA_MAMBA3_BWD_THREADS 64u
#define OA_MAMBA3_BWD_MAXN 32u
#define OA_MAMBA3_BWD_MAXA 8u
#define OA_MAMBA3_BWD_CHUNK 16u
#define OA_MAMBA3_BWD_MAX_WAVES 16u
#define OA_MAMBA3_BWD_SHARED_CHUNK 0
#include "Mamba3SisoBwd.slang"
