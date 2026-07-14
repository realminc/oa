// Gemm — Reusable GEMM (General Matrix Multiply) math functions
//
// Provides tile loading, computation, and storage patterns used across
// all GEMM kernels (naive, tiled, coopmat, stream-k, fused variants).
//
// Import: `import Math;` or `import Gemm;`
// Requires: Storage module for OaLoad/OaStore

module Gemm;

// DTYPE-aware storage — GEMM always computes in FP32 regardless of
// input/output storage format. The accumulator tile is always FP32.
// OaLoad/OaStore respect the DTYPE spec constant (2-byte BF16, 4-byte FP32).
import Storage;

// ============================================================================
// Tile Loading Functions
// ============================================================================

/// Load a tile from matrix A into shared memory with bounds checking
/// @param buf Source buffer (heap[a_idx])
/// @param M Number of rows in A
/// @param K Number of columns in A
/// @param rowBase Starting row of tile in global matrix
/// @param colBase Starting column of tile in global matrix
/// @param tileM Height of tile
/// @param tileK Width of tile
/// @param stride Stride in shared memory (use tileK+1 for bank conflict avoidance)
/// @param smem Shared memory destination
[ForceInline]
public void LoadTileA(
	RWByteAddressBuffer buf,
	uint M, uint K,
	uint rowBase, uint colBase,
	uint tileM, uint tileK,
	uint stride,
	uint tid, uint numThreads,
	out groupshared float smem[])
{
	uint numElements = tileM * tileK;
	for (uint t = tid; t < numElements; t += numThreads) {
		uint r = t / tileK;
		uint c = t % tileK;
		uint globalR = rowBase + r;
		uint globalC = colBase + c;
		smem[r * stride + c] = (globalR < M && globalC < K)
			? OaLoad(buf, globalR * K + globalC)
			: 0.0f;
	}
}

/// Load a tile from matrix B (transposed) into shared memory with bounds checking
/// @param buf Source buffer (heap[b_idx])
/// @param N Number of rows in B (columns in B^T)
/// @param K Number of columns in B (rows in B^T)
/// @param rowBase Starting row of tile in global matrix
/// @param colBase Starting column of tile in global matrix
/// @param tileN Height of tile
/// @param tileK Width of tile
/// @param stride Stride in shared memory (use tileK+1 for bank conflict avoidance)
/// @param smem Shared memory destination
[ForceInline]
public void LoadTileB(
	RWByteAddressBuffer buf,
	uint N, uint K,
	uint rowBase, uint colBase,
	uint tileN, uint tileK,
	uint stride,
	uint tid, uint numThreads,
	out groupshared float smem[])
{
	uint numElements = tileN * tileK;
	for (uint t = tid; t < numElements; t += numThreads) {
		uint r = t / tileK;
		uint c = t % tileK;
		uint globalR = rowBase + r;
		uint globalC = colBase + c;
		smem[r * stride + c] = (globalR < N && globalC < K)
			? OaLoad(buf, globalR * K + globalC)
			: 0.0f;
	}
}

/// Load a tile from matrix A into shared memory (BF16 variant)
/// @param buf Source buffer (heap[a_idx])
/// @param M Number of rows in A
/// @param K Number of columns in A
/// @param rowBase Starting row of tile in global matrix
/// @param colBase Starting column of tile in global matrix
/// @param tileM Height of tile
/// @param tileK Width of tile
/// @param stride Stride in shared memory (use tileK+1 for bank conflict avoidance)
/// @param smem Shared memory destination (BF16)
[ForceInline]
public void LoadTileABf16(
	RWByteAddressBuffer buf,
	uint M, uint K,
	uint rowBase, uint colBase,
	uint tileM, uint tileK,
	uint stride,
	uint tid, uint numThreads,
	out groupshared uint16_t smem[])
{
	uint numElements = tileM * tileK;
	for (uint t = tid; t < numElements; t += numThreads) {
		uint r = t / tileK;
		uint c = t % tileK;
		uint globalR = rowBase + r;
		uint globalC = colBase + c;
		float v = (globalR < M && globalC < K)
			? OaLoad(buf, globalR * K + globalC)
			: 0.0f;
		smem[r * stride + c] = uint16_t(asuint(v) >> 16);  // FP32 → BF16
	}
}

/// Load a tile from matrix B into shared memory (BF16 variant)
[ForceInline]
public void LoadTileBBf16(
	RWByteAddressBuffer buf,
	uint N, uint K,
	uint rowBase, uint colBase,
	uint tileN, uint tileK,
	uint stride,
	uint tid, uint numThreads,
	out groupshared uint16_t smem[])
{
	uint numElements = tileN * tileK;
	for (uint t = tid; t < numElements; t += numThreads) {
		uint r = t / tileK;
		uint c = t % tileK;
		uint globalR = rowBase + r;
		uint globalC = colBase + c;
		float v = (globalR < N && globalC < K)
			? OaLoad(buf, globalR * K + globalC)
			: 0.0f;
		smem[r * stride + c] = uint16_t(asuint(v) >> 16);  // FP32 → BF16
	}
}

// ============================================================================
// Tile Computation Functions
// ============================================================================

/// Compute C += A * B for small register tiles (per-thread computation)
/// @param tileA Local tile from A (TM x K)
/// @param tileB Local tile from B (TN x K)
/// @param acc Accumulator (TM x TN), updated in-place
/// @param TM Height of thread tile
/// @param TN Width of thread tile
/// @param K Inner dimension
[ForceInline]
public void TileMatmulSmall(
	float tileA[], float tileB[],
	inout float acc[],
	uint TM, uint TN, uint K)
{
	for (uint i = 0; i < TM; i++) {
		for (uint j = 0; j < TN; j++) {
			for (uint k = 0; k < K; k++) {
				acc[i * TN + j] += tileA[i * K + k] * tileB[j * K + k];
			}
		}
	}
}

/// Compute C += A * B using shared memory tiles
/// Each thread computes a TM x TN sub-tile
/// @param smA Shared memory tile A (BM x BK_PAD)
/// @param smB Shared memory tile B (BN x BK_PAD)
/// @param acc Thread accumulator (TM x TN)
/// @param threadRow Thread's row in the workgroup grid
/// @param threadCol Thread's column in the workgroup grid
/// @param TM Height of thread tile
/// @param TN Width of thread tile
/// @param BK Inner dimension of block tile
/// @param BK_PAD Padded stride for shared memory
[ForceInline]
public void TileMatmulShared(
	groupshared float smA[],
	groupshared float smB[],
	inout float acc[],
	uint threadRow, uint threadCol,
	uint TM, uint TN, uint BK, uint BK_PAD)
{
	uint aRow = threadRow * TM;
	uint bRow = threadCol * TN;
	
	for (uint k = 0; k < BK; k++) {
		float aVals[16];  // Max TM
		float bVals[16];  // Max TN
		
		for (uint i = 0; i < TM; i++) {
			aVals[i] = smA[(aRow + i) * BK_PAD + k];
		}
		for (uint j = 0; j < TN; j++) {
			bVals[j] = smB[(bRow + j) * BK_PAD + k];
		}
		
		for (uint i = 0; i < TM; i++) {
			for (uint j = 0; j < TN; j++) {
				acc[i * TN + j] += aVals[i] * bVals[j];
			}
		}
	}
}

// ============================================================================
// Tile Storage Functions
// ============================================================================

/// Store a thread's tile to global memory with bounds checking
/// @param buf Destination buffer (heap[out_idx])
/// @param M Number of rows in output
/// @param N Number of columns in output
/// @param rowBase Starting row of workgroup tile
/// @param colBase Starting column of workgroup tile
/// @param threadRow Thread's row in workgroup grid
/// @param threadCol Thread's column in workgroup grid
/// @param acc Thread accumulator (TM x TN)
/// @param TM Height of thread tile
/// @param TN Width of thread tile
[ForceInline]
public void StoreTileC(
	RWByteAddressBuffer buf,
	uint M, uint N,
	uint rowBase, uint colBase,
	uint threadRow, uint threadCol,
	float acc[],
	uint TM, uint TN)
{
	for (uint i = 0; i < TM; i++) {
		for (uint j = 0; j < TN; j++) {
			uint globalR = rowBase + threadRow * TM + i;
			uint globalC = colBase + threadCol * TN + j;
			if (globalR < M && globalC < N) {
				OaStore(buf, globalR * N + globalC, acc[i * TN + j]);
			}
		}
	}
}

/// Store a thread's tile to global memory with accumulation (for gradient accumulation)
/// @param buf Destination buffer (heap[out_idx])
/// @param M Number of rows in output
/// @param N Number of columns in output
/// @param rowBase Starting row of workgroup tile
/// @param colBase Starting column of workgroup tile
/// @param threadRow Thread's row in workgroup grid
/// @param threadCol Thread's column in workgroup grid
/// @param acc Thread accumulator (TM x TN)
/// @param TM Height of thread tile
/// @param TN Width of thread tile
[ForceInline]
public void StoreTileCAccum(
	RWByteAddressBuffer buf,
	uint M, uint N,
	uint rowBase, uint colBase,
	uint threadRow, uint threadCol,
	float acc[],
	uint TM, uint TN)
{
	for (uint i = 0; i < TM; i++) {
		for (uint j = 0; j < TN; j++) {
			uint globalR = rowBase + threadRow * TM + i;
			uint globalC = colBase + threadCol * TN + j;
			if (globalR < M && globalC < N) {
				uint idx = globalR * N + globalC;
				float existing = OaLoad(buf, idx);
				OaStore(buf, idx, existing + acc[i * TN + j]);
			}
		}
	}
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Convert FP32 to BF16 bits (truncate mantissa)
public uint16_t Fp32ToBf16Bits(float v) {
	return uint16_t(asuint(v) >> 16);
}

/// Convert BF16 bits to FP32
public float Bf16BitsToFp32(uint16_t bits) {
	return asfloat(uint(bits) << 16);
}

/// Compute ceil(a / b) for unsigned integers
public uint CeilDiv(uint a, uint b) {
	return (a + b - 1) / b;
}