// Exact short-sequence specialization used by the NLP tutorials. The complete
// P=16, N=32, L=16 state history is 32 KiB and fits in Xe's 48 KiB workgroup
// memory, avoiding the generic kernel's global ChunkBuf round trip.
#define OA_MAMBA3_BWD_THREADS 64u
#define OA_MAMBA3_BWD_MAXN 32u
#define OA_MAMBA3_BWD_MAXA 8u
#define OA_MAMBA3_BWD_CHUNK 16u
#define OA_MAMBA3_BWD_MAX_WAVES 16u
#define OA_MAMBA3_BWD_SHARED_CHUNK 1
#include "Mamba3SisoBwd.slang"
