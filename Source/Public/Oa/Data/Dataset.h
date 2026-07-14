// OA ML - Data Loading
//
// PyTorch torch.utils.data equivalent.
// Byte-first: load raw files as byte sequences. No preprocessing.

#pragma once

#include <Oa/Core/Matrix.h>

// DATASET (torch.utils.data.Dataset)

class OaDataset {
public:
	virtual ~OaDataset() = default;

	/// Multi-tensor sample: input X, target Y (optional), and optional metadata
	struct Sample {
		OaMatrix X;
		OaMatrix Y;  // Optional: empty if dataset has no labels

		Sample() = default;
		explicit Sample(OaMatrix InX) : X(std::move(InX)) {}
		Sample(OaMatrix InX, OaMatrix InY) : X(std::move(InX)), Y(std::move(InY)) {}

		[[nodiscard]] bool HasLabel() const { return !Y.IsEmpty(); }
	};

	/// Number of items
	[[nodiscard]] virtual OaI64 Size() const = 0;

	/// Get single sample by index (unified path)
	[[nodiscard]] virtual Sample GetSample(OaI64 InIndex) const {
		return Sample(GetItem(InIndex));
	}

	/// Get single item by index (backward-compat for single-output datasets)
	[[nodiscard]] virtual OaMatrix GetItem(OaI64 InIndex) const = 0;

	/// Operator[] for convenience
	[[nodiscard]] OaMatrix operator[](OaI64 InIndex) const { return GetItem(InIndex); }
};

// BYTE DATASET - Raw byte sequences from files
// Loads ANY file type as raw bytes. Text, images, audio, binary — all bytes.
// No preprocessing. No tokenization. Just mmap + slice.

class OaByteDataset : public OaDataset {
public:
	/// Load all files from directory as byte sequences
	explicit OaByteDataset(const OaString& InPath, OaI64 InSeqLen = 1024);

	/// Load single file, split into chunks
	OaByteDataset(const OaString& InPath, OaI64 InSeqLen, bool InSingleFile);

	[[nodiscard]] OaI64 Size() const override { return NumSequences_; }
	[[nodiscard]] OaMatrix GetItem(OaI64 InIndex) const override;

	/// Total bytes loaded
	[[nodiscard]] OaI64 TotalBytes() const { return TotalBytes_; }

private:
	OaVec<OaVec<OaU8>> Data_;  // File contents
	OaI64 SeqLen_;
	OaI64 NumSequences_ = 0;
	OaI64 TotalBytes_ = 0;
};

// DATA TRANSFORM — Per-sample or per-batch transform
class OaDataTransform {
public:
	virtual ~OaDataTransform() = default;

	/// Apply in-place to a single sample (called during collation)
	virtual void Apply(OaDataset::Sample& InOutSample) const = 0;
};

// DATA LOADER (torch.utils.data.DataLoader)

class OaDataLoaderConfig {
public:
	OaI32 BatchSize = 32;
	bool Shuffle = true;
	OaU64 Seed = 0;         // 0 = nondeterministic
	OaI32 NumWorkers = 0;    // 0 = main thread
	bool DropLast = false;   // Drop incomplete final batch
	bool PinMemory = false;  // Pin to CPU for faster GPU transfer

	// Per-sample transforms applied before batch assembly
	OaVec<OaSharedPtr<OaDataTransform>> Transforms;
};

class OaDataLoader {
public:
	OaDataLoader(OaDataset& InDataset, OaDataLoaderConfig InConfig = {});

	/// Batched output: X [B, ...] and optional Y [B, ...]
	struct Batch {
		OaMatrix X;
		OaMatrix Y;

		[[nodiscard]] bool IsValid() const { return !X.IsEmpty(); }
		explicit operator bool() const { return IsValid(); }
	};

	/// Get next batch as multi-tensor. Returns nullopt when epoch ends.
	[[nodiscard]] OaOpt<Batch> NextBatch();

	/// Legacy: get next batch into out params. Returns false when epoch complete.
	[[nodiscard]] bool NextBatch(OaMatrix& OutX, OaMatrix& OutY);

	/// Legacy single-tensor path (returns X only). Returns nullopt when epoch ends.
	[[nodiscard]] OaOpt<OaMatrix> Next();

	/// Reset to beginning of epoch (re-shuffles if enabled)
	void Reset();

	/// Number of batches per epoch
	[[nodiscard]] OaI64 NumBatches() const;

	/// Current batch index
	[[nodiscard]] OaI64 CurrentBatch() const { return CurrentBatch_; }

private:
	OaDataset& Dataset_;
	OaDataLoaderConfig Config_;
	OaVec<OaI64> Indices_;
	OaI64 CurrentBatch_ = 0;

	void BuildIndices_();
	void ApplyTransforms_(OaDataset::Sample& InOutSample) const;
};

// MMAP BYTE DATASET - Zero-copy for large files
// Memory-maps the file instead of loading into RAM.
// A 100 GB dataset uses ~0 bytes of process memory (OS pages it on demand).
// Random access into the mmap'd region for training samples.

class OaMMapByteDataset : public OaDataset {
public:
	explicit OaMMapByteDataset(const OaString& InPath, OaI64 InSeqLen = 1024);
	~OaMMapByteDataset() override;

	// Non-copyable, movable
	OaMMapByteDataset(const OaMMapByteDataset&) = delete;
	OaMMapByteDataset& operator=(const OaMMapByteDataset&) = delete;
	OaMMapByteDataset(OaMMapByteDataset&& InOther) noexcept;
	OaMMapByteDataset& operator=(OaMMapByteDataset&& InOther) noexcept;

	[[nodiscard]] OaI64 Size() const override { return NumSequences_; }
	[[nodiscard]] OaMatrix GetItem(OaI64 InIndex) const override;

	/// Total bytes in file
	[[nodiscard]] OaI64 TotalBytes() const { return FileSize_; }

	/// Direct pointer to mmap'd data (for zero-copy batch assembly)
	[[nodiscard]] const OaU8* RawData() const { return MappedData_; }

	/// File size
	[[nodiscard]] OaI64 FileSize() const { return FileSize_; }

	/// Sequence length
	[[nodiscard]] OaI64 SeqLen() const { return SeqLen_; }

private:
	OaU8* MappedData_ = nullptr;
	OaI64 FileSize_ = 0;
	OaI64 SeqLen_ = 0;
	OaI64 NumSequences_ = 0;
	int Fd_ = -1;
};

// PINNED MEMORY (page-locked host memory for fast GPU transfers)
// Pinned memory enables:
//   - Async host-to-device transfers (staging buffer → device-local)
//   - ~2x higher bandwidth vs pageable memory
//   - Overlap of compute and data transfer

/// Allocate pinned (page-locked) host memory
[[nodiscard]] void* OaAllocPinned(OaUsize InBytes);

/// Free pinned memory
void OaFreePinned(void* InPtr);

/// RAII wrapper for pinned host memory
class OaPinnedBuffer {
public:
	OaPinnedBuffer() = default;
	explicit OaPinnedBuffer(OaUsize InBytes);
	~OaPinnedBuffer();

	// Non-copyable, movable
	OaPinnedBuffer(const OaPinnedBuffer&) = delete;
	OaPinnedBuffer& operator=(const OaPinnedBuffer&) = delete;
	OaPinnedBuffer(OaPinnedBuffer&& InOther) noexcept;
	OaPinnedBuffer& operator=(OaPinnedBuffer&& InOther) noexcept;

	[[nodiscard]] void* Data() { return Data_; }
	[[nodiscard]] const void* Data() const { return Data_; }
	[[nodiscard]] OaUsize Size() const { return Size_; }

	/// Typed access
	template<typename T>
	[[nodiscard]] T* As() { return static_cast<T*>(Data_); }

	template<typename T>
	[[nodiscard]] const T* As() const { return static_cast<const T*>(Data_); }

private:
	void* Data_ = nullptr;
	OaUsize Size_ = 0;
};

// GDS BYTE DATASET — GPUDirect Storage (cuFile)
// Reads bytes directly from NVMe into GPU VRAM via cuFile API.
// Works in compat mode (posix fallback) without nvidia_fs kernel module.
// Automatically uses native DMA when nvidia_fs is loaded.

#ifdef OA_USE_GDS

class OaGdsByteDataset : public OaDataset {
public:
	/// Open file for GDS reads. InDevPtr is a pre-allocated GPU buffer
	/// of at least InBufferSize bytes, registered with cuFileBufRegister.
	OaGdsByteDataset(const OaString& InPath, OaI64 InSeqLen, void* InDevPtr, OaUsize InBufferSize);
	~OaGdsByteDataset() override;

	// Non-copyable
	OaGdsByteDataset(const OaGdsByteDataset&) = delete;
	OaGdsByteDataset& operator=(const OaGdsByteDataset&) = delete;

	[[nodiscard]] OaI64 Size() const override { return NumSequences_; }
	[[nodiscard]] OaMatrix GetItem(OaI64 InIndex) const override;

	/// Read raw bytes directly into GPU buffer. Returns bytes read, or -1 on error.
	/// InDevOffset: offset into the registered GPU buffer
	/// InFileOffset: byte offset in the file to read from
	/// InBytes: number of bytes to read
	[[nodiscard]] OaI64 ReadToGPU(OaUsize InDevOffset, OaI64 InFileOffset, OaI64 InBytes) const;

	/// Total bytes in file
	[[nodiscard]] OaI64 TotalBytes() const { return FileSize_; }

	/// Returns true if using native GDS (nvidia_fs), false if compat mode (posix)
	[[nodiscard]] bool IsNativeGDS() const { return IsNative_; }

	/// Check if cuFile driver is available
	[[nodiscard]] static bool IsAvailable();

private:
	void* CuFileHandle_ = nullptr;       // CUfileHandle_t (opaque to avoid cufile.h in header)
	int Fd_ = -1;
	OaI64 FileSize_ = 0;
	OaI64 SeqLen_ = 0;
	OaI64 NumSequences_ = 0;
	void* RegisteredDevPtr_ = nullptr;
	OaUsize RegisteredSize_ = 0;
	bool IsNative_ = false;
};

#endif // OA_USE_GDS

// CONVENIENCE: Create dataset + loader in one call

class OaByteDataPipeline {
public:
	OaUniquePtr<OaByteDataset> Dataset;
	OaUniquePtr<OaDataLoader> Loader;

	static OaByteDataPipeline Create(const OaString& InPath, OaI64 InSeqLen = 1024, OaI32 InBatchSize = 32);
};
