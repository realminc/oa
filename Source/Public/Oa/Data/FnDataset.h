// OaFnDataset — Unified dataset operations (shuffle, split, collate, normalize)
//
// Stateless functional API following the OaFnMatrix / OaFnImage pattern.
// Use OaDataLoader for stateful batch iteration; use OaFnDataset for one-shot
// dataset manipulation (splitting, shuffling, collation).
//
#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/Matrix.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Vision/FnImage.h>
#include <Oa/Data/Dataset.h>

namespace OaFnDataset {

// ============================================================================
// Index shuffling and splitting
// ============================================================================

/// Fisher-Yates shuffle on an index array.
void Shuffle(OaVec<OaI64>& InOutIndices, OaU64 InSeed = 0);

/// Random split result for train / validation / test.
struct SplitResult {
	OaVec<OaI64> Train;
	OaVec<OaI64> Val;
	OaVec<OaI64> Test;
};

/// Randomly split indices into train/val/test subsets.
/// Ratios are clamped so that Train+Val+Test <= TotalSize.
/// Any remainder goes to the training set.
SplitResult RandomSplit(OaI64 InTotalSize,
	OaF32 InTrainRatio = 0.8f,
	OaF32 InValRatio   = 0.1f,
	OaU64 InSeed       = 42);

// ============================================================================
// Batch collation
// ============================================================================

/// Assemble individual samples into batched matrices.
/// X is stacked along dim 0; Y is stacked along dim 0 (if present).
struct Batch {
	OaMatrix X;
	OaMatrix Y;
};

/// Collate a span of samples into a single batch.
/// Returns an empty batch when samples are empty, inconsistent, or cannot be
/// represented with a leading batch dimension.
[[nodiscard]] Batch Collate(OaSpan<const OaDataset::Sample> InSamples);

// ============================================================================
// Normalization & augmentation helpers
// ============================================================================

/// Normalize image batch (or any tensor) using per-channel mean/std.
/// For image batches [B, H, W, C] this delegates to GPU-accelerated
/// vision kernels; for generic tensors it uses OaFnMatrix ops.
[[nodiscard]] OaStatus Normalize(OaMatrix& InOutBatch, const OaNormalizationParams& InParams);

/// Scale batch to a target range [InMin, InMax] (e.g. [0,1] or [-1,1]).
/// For uint8 images this is equivalent to divide-by-255.
[[nodiscard]] OaStatus ScaleToRange(OaMatrix& InOutBatch, OaF32 InMin = 0.0f, OaF32 InMax = 1.0f);

/// Cast batch to a new scalar type (e.g. UInt8 -> Float32).
[[nodiscard]] OaResult<OaMatrix> Cast(const OaMatrix& InBatch, OaScalarType InDtype);

} // namespace OaFnDataset
