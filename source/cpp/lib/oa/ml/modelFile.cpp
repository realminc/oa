#include <oa/core/fnMatrix.h>
#include <oa/core/filesystem.h>
#include <oa/core/log.h>
#include <oa/core/std/algo.h>
#include <oa/core/std/array.h>
#include <oa/core/std/atomic.h>
#include <oa/core/std/chrono.h>
#include <oa/core/std/format.h>
#include <oa/core/std/scalarMath.h>
#include <oa/ml/modelFile.h>
#include <oa/ml/quantMatrixAccess.h>
#include <oa/runtime/engine.h>
#include <oa/runtime/executionSession.h>

#include <stdio.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace oa {

// FNV-1a hash (no external deps — replaces xxhash)
oa::U64 modelFileHash(const oa::U8* inData, oa::Usize inSize) {
	oa::U64 hash = 0xcbf29ce484222325ULL;
	for (oa::Usize i = 0; i < inSize; ++i) {
		hash ^= inData[i];
		hash *= 0x100000001b3ULL;
	}
	return hash;
}

namespace {

constexpr oa::U32 kMaxModelFileSections = 32;
constexpr oa::Usize kHashChunkBytes = 1024 * 1024;

oa::Status modelFileCorrupt(oa::String inMessage) {
	return oa::Status::error(oa::StatusCode::CheckpointCorrupt, "corrupt .oam: " + inMessage);
}

void modelFileHashUpdate(oa::U64& inOutHash, const oa::U8* inData, oa::Usize inSize) {
	for (oa::Usize i = 0; i < inSize; ++i) {
		inOutHash ^= inData[i];
		inOutHash *= 0x100000001b3ULL;
	}
}

oa::U64 modelFileManifestHash(const ModelFileHeader& inHeader, const oa::Vector<ModelFileSectionHeader>& inSections) {
	ModelFileHeader normalized = inHeader;
	normalized.checksum = 0;
	oa::U64 hash = 0xcbf29ce484222325ULL;
	modelFileHashUpdate(hash, reinterpret_cast<const oa::U8*>(&normalized), sizeof(normalized));
	if (not inSections.empty()) {
		modelFileHashUpdate(hash, reinterpret_cast<const oa::U8*>(inSections.data()),
							inSections.size() * sizeof(ModelFileSectionHeader));
	}
	return hash;
}

bool modelFileCheckedAdd(oa::U64 inA, oa::U64 inB, oa::U64& out) {
	if (inB > oa::Limits<oa::U64>::max() - inA)
		return false;
	out = inA + inB;
	return true;
}

bool modelFileCheckedMul(oa::U64 inA, oa::U64 inB, oa::U64& out) {
	if (inA != 0 and inB > oa::Limits<oa::U64>::max() / inA)
		return false;
	out = inA * inB;
	return true;
}

struct ModelFileHandle {
	FILE* value = nullptr;

	ModelFileHandle() = default;
	explicit ModelFileHandle(FILE* inValue) noexcept : value(inValue) {}
	ModelFileHandle(const ModelFileHandle&) = delete;
	ModelFileHandle& operator=(const ModelFileHandle&) = delete;
	~ModelFileHandle() {
		if (value != nullptr) ::fclose(value);
	}

	[[nodiscard]] bool isOpen() const noexcept { return value != nullptr; }

	[[nodiscard]] bool close() noexcept {
		if (value == nullptr) return true;
		FILE* closing = value;
		value = nullptr;
		return ::fclose(closing) == 0;
	}
};

bool modelFileSeek(FILE* inFile, oa::U64 inOffset) {
#ifdef _WIN32
	if (inOffset > static_cast<oa::U64>(oa::Limits<__int64>::max())) return false;
	return ::_fseeki64(inFile, static_cast<__int64>(inOffset), SEEK_SET) == 0;
#else
	if (inOffset > static_cast<oa::U64>(oa::Limits<off_t>::max())) return false;
	return ::fseeko(inFile, static_cast<off_t>(inOffset), SEEK_SET) == 0;
#endif
}

bool modelFileReadExact(FILE* inFile, oa::U64 inOffset, void* outData, oa::U64 inBytes) {
	if (not modelFileSeek(inFile, inOffset)) return false;
	auto* destination = static_cast<oa::U8*>(outData);
	while (inBytes != 0) {
		const oa::Usize chunk = static_cast<oa::Usize>(
			oa::min<oa::U64>(inBytes, oa::Limits<oa::Usize>::max()));
		const oa::Usize read = ::fread(destination, 1, chunk, inFile);
		if (read != chunk) return false;
		destination += read;
		inBytes -= read;
	}
	return true;
}

bool modelFileWriteExact(FILE* inFile, const void* inData, oa::U64 inBytes) {
	const auto* source = static_cast<const oa::U8*>(inData);
	while (inBytes != 0) {
		const oa::Usize chunk = static_cast<oa::Usize>(
			oa::min<oa::U64>(inBytes, oa::Limits<oa::Usize>::max()));
		const oa::Usize written = ::fwrite(source, 1, chunk, inFile);
		if (written != chunk) return false;
		source += written;
		inBytes -= written;
	}
	return true;
}

oa::Result<oa::U64> modelFileHashRange(FILE* inFile, oa::U64 inOffset, oa::U64 inBytes) {
	oa::Vector<oa::U8> chunk(oa::min<oa::U64>(inBytes, kHashChunkBytes));
	oa::U64 hash = 0xcbf29ce484222325ULL;
	oa::U64 consumed = 0;
	while (consumed < inBytes) {
		const oa::U64 bytes = oa::min<oa::U64>(chunk.size(), inBytes - consumed);
		if (not modelFileReadExact(inFile, inOffset + consumed, chunk.data(), bytes)) {
			return modelFileCorrupt("truncated section payload");
		}
		modelFileHashUpdate(hash, chunk.data(), static_cast<oa::Usize>(bytes));
		consumed += bytes;
	}
	return hash;
}

oa::String modelFileTemporaryPath(const oa::String& inFinalPath) {
	static oa::Atomic<oa::U64> sequence{0};
	const auto ticks = oa::steadyNow().nanosecondsSinceEpoch();
	return inFinalPath + ".tmp."
		+ oa::toString(static_cast<oa::I64>(ticks)) + "."
		+ oa::toString(static_cast<oa::I64>(++sequence));
}

oa::Status modelFileAtomicReplace(const oa::String& inTemporaryPath,
								  const oa::String& inFinalPath) {
#ifdef _WIN32
	HANDLE handle = ::CreateFileA(inTemporaryPath.cStr(), GENERIC_READ, FILE_SHARE_READ, nullptr,
								 OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
	if (handle == INVALID_HANDLE_VALUE) {
		return oa::Status::error(oa::StatusCode::Unavailable,
								 "cannot open temporary .oam for durable commit");
	}
	const BOOL flushed = ::FlushFileBuffers(handle);
	::CloseHandle(handle);
	if (not flushed or not ::MoveFileExA(inTemporaryPath.cStr(), inFinalPath.cStr(),
										 MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
		return oa::Status::error(oa::StatusCode::Unavailable, "atomic .oam replacement failed");
	}
#else
	const int fileFd = ::open(inTemporaryPath.cStr(), O_RDONLY | O_CLOEXEC);
	if (fileFd < 0) {
		return oa::Status::error(oa::StatusCode::Unavailable,
								 "cannot open temporary .oam for durable commit");
	}
	const int syncResult = ::fsync(fileFd);
	::close(fileFd);
	if (syncResult != 0 or ::rename(inTemporaryPath.cStr(), inFinalPath.cStr()) != 0) {
		return oa::Status::error(oa::StatusCode::Unavailable, "atomic .oam replacement failed");
	}
	const oa::Path parent = oa::Path(inFinalPath).parentPath();
	const oa::String parentString = parent.empty() ? oa::String(".") : parent.string();
	const int dirFd = ::open(parentString.cStr(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
	if (dirFd >= 0) {
		(void)::fsync(dirFd);
		::close(dirFd);
	}
#endif
	return oa::Status::ok();
}

bool modelFileHasTerminator(const char* inData, oa::Usize inSize) {
	for (oa::Usize index = 0; index < inSize; ++index) {
		if (inData[index] == '\0') return true;
	}
	return false;
}

bool modelFileValidScalarType(oa::ScalarType inType) {
	return static_cast<oa::U8>(inType) <= static_cast<oa::U8>(oa::ScalarType::Complex128);
}

bool modelFileReservedBytesAreZero(const ModelTensorEntry& inEntry) {
	for (const oa::U8 value : inEntry.reserved) {
		if (value != 0)
			return false;
	}
	return true;
}

bool modelFileCheckedElementCount(const ModelTensorEntry& inEntry, oa::U64& out) {
	if (inEntry.rank == 0) {
		// Preserve the v1/v2 scalar convention: rank zero encodes one value.
		out = 1;
		return true;
	}
	out = 1;
	for (oa::U8 dim = 0; dim < inEntry.rank; ++dim) {
		if (not modelFileCheckedMul(out, inEntry.shape[dim], out))
			return false;
	}
	return true;
}

bool modelFileQuantizedLayout(const ModelTensorEntry& inEntry, oa::U64& outPayloadBytes,
							  oa::U64& outScaleBytes, oa::U64& outElements) {
	if (inEntry.blockSize != 32 or
		(inEntry.encoding != ModelTensorEncoding::Q4 and
		 inEntry.encoding != ModelTensorEncoding::Q8) or
		not modelFileCheckedElementCount(inEntry, outElements)) {
		return false;
	}
	const oa::U64 blocks = outElements / 32 + (outElements % 32 != 0 ? 1 : 0);
	const oa::U64 payloadPerBlock = inEntry.encoding == ModelTensorEncoding::Q4 ? 16 : 32;
	return modelFileCheckedMul(blocks, payloadPerBlock, outPayloadBytes) and
		   modelFileCheckedMul(blocks, sizeof(oa::F32), outScaleBytes);
}

oa::Result<oa::U64> modelFileExpectedTensorBytes(const ModelTensorEntry& inEntry, bool inIsWeight,
												 oa::U32 inFormatVersion) {
	if (not modelFileReservedBytesAreZero(inEntry)) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
								 "tensor reserved bytes are nonzero");
	}
	if (inEntry.encoding == ModelTensorEncoding::Dense) {
		if (inEntry.blockSize != 0) {
			return oa::Status::error(oa::StatusCode::InvalidArgument,
									 "dense tensor has a nonzero block size");
		}
		oa::U64 elements = 0;
		oa::U64 expected = 0;
		if (not modelFileCheckedElementCount(inEntry, elements) or
			not modelFileCheckedMul(elements, oa::scalarSize(inEntry.dtype), expected)) {
			return oa::Status::error(oa::StatusCode::InvalidArgument,
									 "dense tensor byte count overflow");
		}
		return expected;
	}
	if (inFormatVersion < 3) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
								 "quantized tensor encoding requires OAM v3");
	}
	if (not inIsWeight) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
								 "quantized state tensors are not admitted");
	}
	if (inEntry.dtype != oa::ScalarType::Float32) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
								 "quantized tensor logical dtype must be Float32");
	}
	oa::U64 payloadBytes = 0;
	oa::U64 scaleBytes = 0;
	oa::U64 elements = 0;
	oa::U64 expected = 0;
	if (not modelFileQuantizedLayout(inEntry, payloadBytes, scaleBytes, elements) or
		elements == 0 or not modelFileCheckedAdd(payloadBytes, scaleBytes, expected)) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
								 "invalid quantized tensor layout");
	}
	return expected;
}

oa::I32 modelFileRoundNearestEven(oa::F32 inValue) {
	const oa::F32 lowerValue = oa::floor(inValue);
	const auto lower = static_cast<oa::I32>(lowerValue);
	const oa::F32 fraction = inValue - lowerValue;
	if (fraction < 0.5F)
		return lower;
	if (fraction > 0.5F)
		return lower + 1;
	return oa::abs(lower) % 2 == 0 ? lower : lower + 1;
}

oa::Result<oa::Vector<oa::U8>> modelFileQuantizeFloat32(const oa::U8* inData, oa::U64 inElements,
													 oa::Quantization inQuantization) {
	if (inQuantization != oa::Quantization::Q4 and inQuantization != oa::Quantization::Q8) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "unsupported quantization value");
	}
	const oa::U64 blocks = inElements / 32 + (inElements % 32 != 0 ? 1 : 0);
	const oa::U64 payloadPerBlock = inQuantization == oa::Quantization::Q4 ? 16 : 32;
	oa::U64 payloadBytes = 0;
	oa::U64 scaleBytes = 0;
	oa::U64 totalBytes = 0;
	if (not modelFileCheckedMul(blocks, payloadPerBlock, payloadBytes) or
		not modelFileCheckedMul(blocks, sizeof(oa::F32), scaleBytes) or
		not modelFileCheckedAdd(payloadBytes, scaleBytes, totalBytes) or
		totalBytes > oa::Limits<oa::Usize>::max()) {
		return oa::Status::error(oa::StatusCode::ResourceExhausted,
								 "quantized weight size overflow");
	}
	oa::Vector<oa::U8> encoded(static_cast<oa::Usize>(totalBytes), 0);
	for (oa::U64 block = 0; block < blocks; ++block) {
		oa::F32 maximum = 0.0F;
		for (oa::U64 lane = 0; lane < 32; ++lane) {
			const oa::U64 index = block * 32 + lane;
			if (index >= inElements)
				break;
			oa::F32 value = 0.0F;
			oa::memcpy(&value, inData + index * sizeof(oa::F32), sizeof(value));
			if (oa::isFinite(value))
				maximum = oa::max(maximum, oa::abs(value));
		}
		const oa::F32 divisor = inQuantization == oa::Quantization::Q4 ? 7.0F : 127.0F;
		oa::F32 scale = maximum / divisor;
		if (not(scale > 0.0F) or not oa::isFinite(scale))
			scale = 1.0F;
		oa::memcpy(encoded.data() + payloadBytes + block * sizeof(oa::F32), &scale, sizeof(scale));

		for (oa::U64 lane = 0; lane < 32; ++lane) {
			const oa::U64 index = block * 32 + lane;
			oa::F32 value = 0.0F;
			if (index < inElements) {
				oa::memcpy(&value, inData + index * sizeof(oa::F32), sizeof(value));
				if (not oa::isFinite(value))
					value = 0.0F;
			}
			const oa::I32 limit = inQuantization == oa::Quantization::Q4 ? 7 : 127;
			const oa::I32 quantized =
				oa::clamp(modelFileRoundNearestEven(value / scale), -limit, limit);
			if (inQuantization == oa::Quantization::Q4) {
				const oa::U64 byte = block * 16 + lane / 2;
				const oa::U8 nibble = static_cast<oa::U8>(quantized + 7);
				encoded[static_cast<oa::Usize>(byte)] |=
					static_cast<oa::U8>(nibble << ((lane % 2) * 4));
			} else {
				encoded[static_cast<oa::Usize>(block * 32 + lane)] =
					static_cast<oa::U8>(static_cast<oa::U32>(quantized) & 0xFFU);
			}
		}
	}
	return encoded;
}

} // namespace

// Name lookups
const char* modelFileSectionName(ModelFileSection inType) {
	switch (inType) {
	case ModelFileSection::Config:
		return "Config";
	case ModelFileSection::Weights:
		return "Weights";
	case ModelFileSection::State:
		return "State";
	case ModelFileSection::Optimizer:
		return "Optimizer";
	case ModelFileSection::Progress:
		return "Progress";
	case ModelFileSection::LegacyKernelCache:
		return "LegacyKernelCache";
	default:
		return "Unknown";
	}
}

const char* modelFileTensorEncodingName(ModelTensorEncoding inEncoding) {
	switch (inEncoding) {
	case ModelTensorEncoding::Dense:
		return "Dense";
	case ModelTensorEncoding::Q4:
		return "Q4";
	case ModelTensorEncoding::Q8:
		return "Q8";
	default:
		return "Unknown";
	}
}

// ModelFile methods
const ModelTensorEntry* ModelFile::findWeight(const char* inName) const {
	for (const auto& e : weightIndex) {
		if (oa::strncmp(e.name, inName, kModelFileMaxName) == 0) {
			return &e;
		}
	}
	return nullptr;
}

const void* ModelFile::weightPtr(const char* inName) const {
	const auto* entry = findWeight(inName);
	if (!entry || weightBlob.empty()) {
		return nullptr;
	}
	return weightBlob.data() + entry->blobOffset;
}

const ModelTensorEntry* ModelFile::findState(const char* inName) const {
	for (const auto& e : stateIndex) {
		if (oa::strncmp(e.name, inName, kModelFileMaxName) == 0) {
			return &e;
		}
	}
	return nullptr;
}

const void* ModelFile::statePtr(const char* inName) const {
	const auto* entry = findState(inName);
	if (!entry || stateBlob.empty()) {
		return nullptr;
	}
	return stateBlob.data() + entry->blobOffset;
}

void* ModelFile::statePtr(const char* inName) {
	return const_cast<void*>(static_cast<const ModelFile&>(*this).statePtr(inName));
}

static void addTensor(oa::Vector<ModelTensorEntry>& outIndex, oa::Vector<oa::U8>& outBlob,
					  const char* inName, oa::ScalarType inDtype, oa::Span<const oa::U64> inShape,
					  const void* inData, oa::U64 inBytes,
					  ModelTensorEncoding inEncoding = ModelTensorEncoding::Dense,
					  oa::U8 inBlockSize = 0) {
	OA_ASSERT(inShape.size() <= kModelFileMaxRank);
	OA_ASSERT(inData != nullptr || inBytes == 0);
	ModelTensorEntry entry;
	for (oa::Usize index = 0; index + 1U < kModelFileMaxName and inName[index] != '\0'; ++index) {
		entry.name[index] = inName[index];
	}
	entry.blobOffset = outBlob.size();
	entry.numBytes = inBytes;
	entry.dtype = inDtype;
	entry.rank = static_cast<oa::U8>(inShape.size());
	entry.encoding = inEncoding;
	entry.blockSize = inBlockSize;
	for (oa::Usize i = 0; i < inShape.size() && i < kModelFileMaxRank; ++i)
		entry.shape[i] = inShape[i];
	outIndex.pushBack(entry);

	oa::Usize oldSize = outBlob.size();
	outBlob.resize(oldSize + inBytes);
	if (inBytes > 0) {
		oa::memcpy(outBlob.data() + oldSize, inData, inBytes);
	}
}

void ModelFile::addWeight(const char* inName, oa::ScalarType inDtype,
						  oa::Span<const oa::U64> inShape, const void* inData, oa::U64 inBytes) {
	addTensor(weightIndex, weightBlob, inName, inDtype, inShape, inData, inBytes);
}

void ModelFile::addState(const char* inName, oa::ScalarType inDtype,
						 oa::Span<const oa::U64> inShape, const void* inData, oa::U64 inBytes) {
	addTensor(stateIndex, stateBlob, inName, inDtype, inShape, inData, inBytes);
}

oa::Result<ModelFile> ModelFile::quantizeWeights(oa::Quantization inQuantization) const {
	if (inQuantization != oa::Quantization::Q4 and inQuantization != oa::Quantization::Q8) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
								 "unsupported OAM quantization value");
	}
	ModelFile output = *this;
	output.formatVersion = kModelFileVersion;
	output.weightIndex.clear();
	output.weightBlob.clear();
	output.optimizer = {};
	output.adamM.clear();
	output.adamV.clear();
	output.optimizerPresent = false;

	oa::Usize converted = 0;
	for (const auto& entry : weightIndex) {
		auto expected = modelFileExpectedTensorBytes(entry, true, formatVersion);
		if (not expected.isOk() or expected.getValue() != entry.numBytes) {
			return oa::Status::error(oa::StatusCode::InvalidArgument,
									 oa::String("invalid source weight '") + entry.name + "'");
		}
		oa::U64 entryEnd = 0;
		if (not modelFileCheckedAdd(entry.blobOffset, entry.numBytes, entryEnd) or
			entryEnd > weightBlob.size()) {
			return oa::Status::error(oa::StatusCode::InvalidArgument,
									 oa::String("source weight payload is out of range: ") +
										 entry.name);
		}
		if (entry.encoding != ModelTensorEncoding::Dense) {
			return oa::Status::error(oa::StatusCode::FailedPrecondition,
									 oa::String("source already contains a quantized weight: ") +
										 entry.name);
		}
		const auto* data = weightBlob.data() + entry.blobOffset;
		const oa::Span<const oa::U64> shape(entry.shape, entry.rank);
		if (entry.rank < 2 or entry.numBytes == 0) {
			addTensor(output.weightIndex, output.weightBlob, entry.name, entry.dtype, shape, data,
					  entry.numBytes);
			continue;
		}
		if (entry.dtype != oa::ScalarType::Float32) {
			return oa::Status::error(oa::StatusCode::DtypeMismatch,
									 oa::String("matrix weight is not Float32: ") + entry.name);
		}
		oa::U64 elements = 0;
		if (not modelFileCheckedElementCount(entry, elements) or elements == 0) {
			return oa::Status::error(oa::StatusCode::InvalidArgument,
									 oa::String("matrix weight has an invalid shape: ") +
										 entry.name);
		}
		auto encoded = modelFileQuantizeFloat32(data, elements, inQuantization);
		if (not encoded.isOk())
			return encoded.getStatus();
		const ModelTensorEncoding encoding = inQuantization == oa::Quantization::Q4
												 ? ModelTensorEncoding::Q4
												 : ModelTensorEncoding::Q8;
		addTensor(output.weightIndex, output.weightBlob, entry.name, oa::ScalarType::Float32, shape,
				  encoded->data(), encoded->size(), encoding, 32);
		++converted;
	}
	if (converted == 0) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
								 "model contains no non-empty rank-2-or-higher Float32 weights");
	}
	return output;
}

oa::Result<oa::QuantMatrix> ModelFile::loadQuantMatrix(oa::Engine& inEngine,
													   const char* inName) const {
	const ModelTensorEntry* entry = findWeight(inName);
	if (entry == nullptr) {
		return oa::Status::notFound(oa::String("quantized weight not found: ") + inName);
	}
	if (entry->encoding == ModelTensorEncoding::Dense) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
								 oa::String("weight is dense: ") + inName);
	}
	auto expected = modelFileExpectedTensorBytes(*entry, true, formatVersion);
	if (not expected.isOk() or expected.getValue() != entry->numBytes) {
		return oa::Status::error(oa::StatusCode::CheckpointCorrupt,
								 oa::String("invalid quantized weight layout: ") + inName);
	}
	oa::U64 entryEnd = 0;
	if (not modelFileCheckedAdd(entry->blobOffset, entry->numBytes, entryEnd) or
		entryEnd > weightBlob.size()) {
		return oa::Status::error(oa::StatusCode::CheckpointCorrupt,
								 oa::String("quantized weight payload is out of range: ") + inName);
	}
	oa::U64 payloadBytes = 0;
	oa::U64 scaleBytes = 0;
	oa::U64 elements = 0;
	if (not modelFileQuantizedLayout(*entry, payloadBytes, scaleBytes, elements) or
		payloadBytes > static_cast<oa::U64>(oa::Limits<oa::I64>::max()) or
		scaleBytes / sizeof(oa::F32) > static_cast<oa::U64>(oa::Limits<oa::I64>::max())) {
		return oa::Status::error(oa::StatusCode::CheckpointCorrupt,
								 oa::String("quantized weight dimensions are not representable: ") +
									 inName);
	}
	oa::MatrixShape logicalShape;
	logicalShape.rank = entry->rank;
	for (oa::U8 dim = 0; dim < entry->rank; ++dim) {
		if (entry->shape[dim] > static_cast<oa::U64>(oa::Limits<oa::I64>::max())) {
			return oa::Status::error(oa::StatusCode::CheckpointCorrupt,
									 oa::String("quantized weight shape is not representable: ") +
										 inName);
		}
		logicalShape[dim] = static_cast<oa::I64>(entry->shape[dim]);
	}
	const auto* encoded = weightBlob.data() + entry->blobOffset;
	auto& context = oa::ExecutionSession::forEngine(inEngine);
	oa::ExecutionSession::RecordingScope recording(context);
	auto payload = oa::FnMatrix::fromBytes(
		{encoded, static_cast<oa::Usize>(payloadBytes)},
		oa::MatrixShape{static_cast<oa::I64>(payloadBytes)},
		entry->encoding == ModelTensorEncoding::Q4 ? oa::ScalarType::UInt8 : oa::ScalarType::Int8);
	auto scale =
		oa::FnMatrix::fromBytes({encoded + payloadBytes, static_cast<oa::Usize>(scaleBytes)},
								oa::MatrixShape{static_cast<oa::I64>(scaleBytes / sizeof(oa::F32))},
								oa::ScalarType::Float32);
	if (payload.isEmpty() or scale.isEmpty()) {
		return oa::Status::error(oa::StatusCode::Internal,
								 oa::String("failed to upload quantized weight: ") + inName);
	}
	return oa::QuantMatrixAccess::make(
		oa::move(payload), oa::move(scale), logicalShape,
		entry->encoding == ModelTensorEncoding::Q4 ? oa::Quantization::Q4 : oa::Quantization::Q8);
}

// Tensor index serialization

static oa::Vector<oa::U8> serializeTensorIndex(const oa::Vector<ModelTensorEntry>& inIndex,
											const oa::Vector<oa::U8>& inBlob) {
	oa::U32 count = static_cast<oa::U32>(inIndex.size());
	oa::U32 reserved = 0;
	oa::Usize indexBytes = sizeof(oa::U32) * 2 + sizeof(ModelTensorEntry) * count;

	oa::Vector<oa::U8> raw(indexBytes + inBlob.size());
	oa::U8* p = raw.data();
	oa::memcpy(p, &count, sizeof(oa::U32));
	p += sizeof(oa::U32);
	oa::memcpy(p, &reserved, sizeof(oa::U32));
	p += sizeof(oa::U32);
	for (const auto& e : inIndex) {
		oa::memcpy(p, &e, sizeof(ModelTensorEntry));
		p += sizeof(ModelTensorEntry);
	}
	oa::memcpy(p, inBlob.data(), inBlob.size());
	return raw;
}

// save

oa::Status ModelFile::save(const oa::String& inPath) const {
	const oa::Path parent = oa::Path(inPath).parentPath();
	if (not parent.empty()) OA_RETURN_IF_ERROR(oa::Filesystem::createDirectories(parent));
	auto validateTensors = [&](const oa::Vector<ModelTensorEntry>& inIndex,
							   const oa::Vector<oa::U8>& inBlob, bool inIsWeight) -> oa::Status {
		oa::HashSet<oa::String> names;
		oa::Vector<oa::Pair<oa::U64, oa::U64>> tensorRanges;
		for (const auto& entry : inIndex) {
			if (not modelFileHasTerminator(entry.name, sizeof(entry.name)) or
				entry.name[0] == '\0' or entry.rank > kModelFileMaxRank or
				not modelFileValidScalarType(entry.dtype) or
				not names.insert(oa::String(entry.name)).second) {
				return oa::Status::error(oa::StatusCode::InvalidArgument,
										 "cannot save .oam with invalid tensor metadata");
			}
			auto expected = modelFileExpectedTensorBytes(entry, inIsWeight, kModelFileVersion);
			if (not expected.isOk() or expected.getValue() != entry.numBytes) {
				return oa::Status::error(
					oa::StatusCode::InvalidArgument,
					oa::String("cannot save .oam with invalid tensor layout: ") + entry.name);
			}
			oa::U64 end = 0;
			if (not modelFileCheckedAdd(entry.blobOffset, entry.numBytes, end) or
				end > inBlob.size()) {
				return oa::Status::error(oa::StatusCode::InvalidArgument,
										 oa::String("cannot save .oam with out-of-range tensor: ") +
											 entry.name);
			}
			if (entry.numBytes != 0)
				tensorRanges.emplaceBack(entry.blobOffset, end);
		}
		oa::sort(tensorRanges.begin(), tensorRanges.end(),
			[](const auto& inLeft, const auto& inRight) {
				return inLeft.first < inRight.first;
			});
		for (oa::Usize i = 1; i < tensorRanges.size(); ++i) {
			if (tensorRanges[i].first < tensorRanges[i - 1].second) {
				return oa::Status::error(oa::StatusCode::InvalidArgument, "cannot save .oam with overlapping tensor payload ranges");
			}
		}
		return oa::Status::ok();
	};
	OA_RETURN_IF_ERROR(validateTensors(weightIndex, weightBlob, true));
	OA_RETURN_IF_ERROR(validateTensors(stateIndex, stateBlob, false));

	struct Payload {
		ModelFileSection type;
		oa::Vector<oa::U8> data;
	};
	oa::Vector<Payload> payloads;

	// Config + optional archConfig
	{
		oa::Vector<oa::U8> raw(sizeof(ModelFileConfig) + archConfig.size());
		oa::memcpy(raw.data(), &config, sizeof(ModelFileConfig));
		if (!archConfig.empty())
			oa::memcpy(raw.data() + sizeof(ModelFileConfig), archConfig.data(), archConfig.size());
		payloads.pushBack({ModelFileSection::Config, oa::move(raw)});
	}

	if (hasWeights()) {
		payloads.pushBack(
			{ModelFileSection::Weights, serializeTensorIndex(weightIndex, weightBlob)});
	}

	if (hasState()) {
		payloads.pushBack({ModelFileSection::State, serializeTensorIndex(stateIndex, stateBlob)});
	}

	if (hasOptimizer()) {
		if (not modelFileHasKnownOptimizer(optimizer)) {
			return oa::Status::error(oa::StatusCode::InvalidArgument,
									 "cannot save .oam with an unknown optimizer type");
		}
		ModelOptimizerState hdr = optimizer;
		hdr.numParams = adamM.size();
		const oa::Usize adamMBytes = adamM.size() * sizeof(oa::F32);
		const oa::Usize adamVBytes = adamV.size() * sizeof(oa::F32);
		oa::Vector<oa::U8> raw(sizeof(ModelOptimizerState) + adamMBytes + adamVBytes);
		oa::U8* p = raw.data();
		oa::memcpy(p, &hdr, sizeof(ModelOptimizerState));
		p += sizeof(ModelOptimizerState);
		if (adamMBytes > 0) {
			oa::memcpy(p, adamM.data(), adamMBytes);
			p += adamMBytes;
		}
		if (adamVBytes > 0) {
			oa::memcpy(p, adamV.data(), adamVBytes);
			p += adamVBytes;
		}
		payloads.pushBack({ModelFileSection::Optimizer, oa::move(raw)});
	}

	{
		oa::Vector<oa::U8> raw(sizeof(ModelTrainingProgress));
		oa::memcpy(raw.data(), &progress, sizeof(ModelTrainingProgress));
		payloads.pushBack({ModelFileSection::Progress, oa::move(raw)});
	}

	oa::U32 numSections = static_cast<oa::U32>(payloads.size());
	oa::Usize headerBytes = kModelFileHeaderSize + numSections * kModelFileSectionHeaderSize;
	oa::Usize dataStart = modelFilePageAlign(headerBytes);

	oa::Vector<ModelFileSectionHeader> sectionHeaders(numSections);
	oa::U64 offset = dataStart;

	for (oa::U32 i = 0; i < numSections; ++i) {
		auto& sh = sectionHeaders[i];
		auto& sp = payloads[i];
		sh.type = static_cast<oa::U32>(sp.type);
		sh.compression = ModelFileCompression::None;
		sh.offset = offset;
		sh.size = sp.data.size();
		sh.compressedSize = 0;
		sh.checksum = modelFileHash(sp.data.data(), sp.data.size());

		bool needsAlign =
			(sp.type == ModelFileSection::Weights || sp.type == ModelFileSection::State);
		offset += needsAlign ? modelFilePageAlign(sp.data.size()) : sp.data.size();
	}

	ModelFileHeader fileHeader;
	fileHeader.numSections = numSections;
	fileHeader.totalSize = offset;
	fileHeader.checksum = modelFileManifestHash(fileHeader, sectionHeaders);

	const oa::String temporaryPath = modelFileTemporaryPath(inPath);
	ModelFileHandle file(::fopen(temporaryPath.cStr(), "wb"));
	if (not file.isOpen())
		return oa::Status::error("Failed to open for write: " + temporaryPath);

	bool complete = modelFileWriteExact(file.value, &fileHeader, sizeof(fileHeader));
	for (const auto& sh : sectionHeaders) {
		complete = complete and modelFileWriteExact(file.value, &sh, sizeof(sh));
	}

	{
		oa::Usize written = kModelFileHeaderSize + numSections * kModelFileSectionHeaderSize;
		oa::Vector<oa::U8> zeros(dataStart - written, 0);
		complete = complete and modelFileWriteExact(file.value, zeros.data(), zeros.size());
	}

	for (oa::U32 i = 0; i < numSections; ++i) {
		const auto& sp = payloads[i];
		complete = complete and modelFileWriteExact(file.value, sp.data.data(), sp.data.size());
		bool needsAlign =
			(sp.type == ModelFileSection::Weights || sp.type == ModelFileSection::State);
		if (needsAlign) {
			oa::Usize pad = modelFilePageAlign(sp.data.size()) - sp.data.size();
			if (pad > 0) {
				oa::Vector<oa::U8> zeros(pad, 0);
				complete = complete and modelFileWriteExact(file.value, zeros.data(), pad);
			}
		}
	}
	complete = complete and ::fflush(file.value) == 0;
	if (not complete) {
		(void)file.close();
		(void)oa::Filesystem::removeFile(oa::Path(temporaryPath));
		return oa::Status::error(oa::StatusCode::DiskFull,
								 "failed to write complete .oam: " + inPath);
	}
	if (not file.close()) {
		(void)oa::Filesystem::removeFile(oa::Path(temporaryPath));
		return oa::Status::error(oa::StatusCode::Unavailable,
								 "failed to close .oam temporary file: " + inPath);
	}
	const auto commitStatus = modelFileAtomicReplace(temporaryPath, inPath);
	if (not commitStatus.isOk()) {
		(void)oa::Filesystem::removeFile(oa::Path(temporaryPath));
		return commitStatus;
	}

	OaLogInfo(oa::LogComponent::Ml, "[oam] Saved {} | {} weights | {:.1f} MB", inPath.cStr(),
			  weightIndex.size(), fileHeader.totalSize / 1e6);
	return oa::Status::ok();
}

// load

oa::Result<ModelFile> ModelFile::load(const oa::String& inPath) {
	ModelFileHandle file(::fopen(inPath.cStr(), "rb"));
	if (not file.isOpen())
		return oa::Status::error("Cannot open: " + inPath);

	auto sizeResult = oa::Filesystem::getFileSize(oa::Path(inPath));
	if (not sizeResult.isOk()) return sizeResult.getStatus();
	const oa::U64 fileSize = sizeResult.getValue();
	if (fileSize < sizeof(ModelFileHeader))
		return modelFileCorrupt("truncated file header");

	ModelFileHeader fh;
	if (not modelFileReadExact(file.value, 0, &fh, sizeof(fh))) {
		return modelFileCorrupt("failed to read file header");
	}
	if (fh.magic != kModelFileMagic)
		return modelFileCorrupt("invalid magic");
	if (fh.version < kModelFileMinVersion or fh.version > kModelFileVersion) {
		return modelFileCorrupt("unsupported format version");
	}
	if (fh.numSections == 0 or fh.numSections > kMaxModelFileSections) {
		return modelFileCorrupt("invalid section count");
	}
	oa::U64 sectionTableBytes = 0;
	if (not modelFileCheckedMul(fh.numSections, sizeof(ModelFileSectionHeader),
								sectionTableBytes)) {
		return modelFileCorrupt("section table size overflow");
	}
	oa::U64 headerBytes = 0;
	if (not modelFileCheckedAdd(sizeof(ModelFileHeader), sectionTableBytes, headerBytes) or
		headerBytes > fileSize) {
		return modelFileCorrupt("truncated section table");
	}
	if (fh.totalSize != fileSize)
		return modelFileCorrupt("file size does not match header");

	oa::Vector<ModelFileSectionHeader> sections(fh.numSections);
	if (not modelFileReadExact(file.value, sizeof(ModelFileHeader), sections.data(), sectionTableBytes)) {
		return modelFileCorrupt("failed to read section table");
	}
	if (fh.version >= 2 and fh.checksum != modelFileManifestHash(fh, sections)) {
		return modelFileCorrupt("file metadata checksum mismatch");
	}

	const oa::U64 dataStart = modelFilePageAlign(static_cast<oa::Usize>(headerBytes));
	oa::Array<bool, static_cast<oa::Usize>(ModelFileSection::LegacyKernelCache) + 1> seen{};
	oa::Vector<oa::Pair<oa::U64, oa::U64>> ranges;
	oa::U64 legacyFileChecksum = 0;
	for (const auto& sh : sections) {
		if (sh.type < static_cast<oa::U32>(ModelFileSection::Config) or
			sh.type > static_cast<oa::U32>(ModelFileSection::LegacyKernelCache)) {
			return modelFileCorrupt("unknown section type");
		}
		if (seen[sh.type])
			return modelFileCorrupt("duplicate section type");
		seen[sh.type] = true;
		if (sh.compression != ModelFileCompression::None or sh.compressedSize != 0) {
			return modelFileCorrupt("unsupported section compression");
		}
		oa::U64 sectionEnd = 0;
		if (sh.offset < dataStart or not modelFileCheckedAdd(sh.offset, sh.size, sectionEnd) or
			sectionEnd > fileSize) {
			return modelFileCorrupt("section range is outside the file");
		}
		// ModelFileSectionHeader is packed on disk.  Do not forward references to
		// its 64-bit members into standard-library constructors: those references
		// would retain the packed, potentially misaligned address.
		const oa::U64 sectionOffset = sh.offset;
		ranges.emplaceBack(sectionOffset, sectionEnd);
		auto hash = modelFileHashRange(file.value, sh.offset, sh.size);
		if (not hash.isOk())
			return hash.getStatus();
		if (hash.getValue() != sh.checksum) {
			return modelFileCorrupt("section payload checksum mismatch");
		}
		legacyFileChecksum ^= hash.getValue();
	}
	oa::sort(ranges.begin(), ranges.end(),
		[](const auto& inLeft, const auto& inRight) {
			return inLeft.first < inRight.first;
		});
	for (oa::Usize i = 1; i < ranges.size(); ++i) {
		if (ranges[i].first < ranges[i - 1].second) {
			return modelFileCorrupt("overlapping section ranges");
		}
	}
	if (fh.version == 1 and legacyFileChecksum != fh.checksum) {
		return modelFileCorrupt("legacy file checksum mismatch");
	}
	if (not seen[static_cast<oa::Usize>(ModelFileSection::Config)] or
		not seen[static_cast<oa::Usize>(ModelFileSection::Progress)]) {
		return modelFileCorrupt("required section is missing");
	}

	auto findSection = [&](ModelFileSection inType) -> const ModelFileSectionHeader* {
		for (const auto& sh : sections) {
			if (sh.type == static_cast<oa::U32>(inType))
				return &sh;
		}
		return nullptr;
	};

	ModelFile model;
	model.formatVersion = fh.version;
	const auto* configSection = findSection(ModelFileSection::Config);
	if (configSection->size < sizeof(ModelFileConfig) or
		not modelFileReadExact(file.value, configSection->offset, &model.config,
							   sizeof(ModelFileConfig))) {
		return modelFileCorrupt("invalid config section");
	}
	oa::U64 expectedConfigSize = 0;
	if (not modelFileCheckedAdd(sizeof(ModelFileConfig), model.config.archConfigSize,
								expectedConfigSize) or
		expectedConfigSize != configSection->size) {
		return modelFileCorrupt("architecture config size mismatch");
	}
	if (not modelFileHasTerminator(model.config.architecture, sizeof(model.config.architecture)) or
		not modelFileValidScalarType(static_cast<oa::ScalarType>(model.config.weightDtype)) or
		not modelFileValidScalarType(static_cast<oa::ScalarType>(model.config.stateDtype)) or
		not modelFileValidScalarType(static_cast<oa::ScalarType>(model.config.computeDtype))) {
		return modelFileCorrupt("invalid config metadata");
	}
	if (model.config.archConfigSize > 0) {
		model.archConfig.resize(model.config.archConfigSize);
		if (not modelFileReadExact(file.value, configSection->offset + sizeof(ModelFileConfig),
								   model.archConfig.data(), model.archConfig.size())) {
			return modelFileCorrupt("truncated architecture config");
		}
	}

	auto parseTensorSection = [&](ModelFileSection inType, oa::Vector<ModelTensorEntry>& outIndex,
								  oa::Vector<oa::U8>& outBlob) -> oa::Status {
		const auto* sh = findSection(inType);
		if (sh == nullptr)
			return oa::Status::ok();
		if (sh->size < sizeof(oa::U32) * 2)
			return modelFileCorrupt("truncated tensor index");
		oa::U32 countAndReserved[2]{};
		if (not modelFileReadExact(file.value, sh->offset, countAndReserved, sizeof(countAndReserved))) {
			return modelFileCorrupt("cannot read tensor index header");
		}
		if (countAndReserved[1] != 0)
			return modelFileCorrupt("tensor index reserved field is nonzero");
		oa::U64 entriesBytes = 0;
		oa::U64 indexBytes = 0;
		if (not modelFileCheckedMul(countAndReserved[0], sizeof(ModelTensorEntry), entriesBytes) or
			not modelFileCheckedAdd(sizeof(countAndReserved), entriesBytes, indexBytes) or
			indexBytes > sh->size) {
			return modelFileCorrupt("tensor index size overflow");
		}
		outIndex.resize(countAndReserved[0]);
		if (entriesBytes > 0 and not modelFileReadExact(file.value, sh->offset + sizeof(countAndReserved),
														outIndex.data(), entriesBytes)) {
			return modelFileCorrupt("truncated tensor index entries");
		}
		const oa::U64 blobSize = sh->size - indexBytes;
		oa::HashSet<oa::String> names;
		oa::Vector<oa::Pair<oa::U64, oa::U64>> tensorRanges;
		for (const auto& entry : outIndex) {
			if (not modelFileHasTerminator(entry.name, sizeof(entry.name)) or
				entry.name[0] == '\0' or entry.rank > kModelFileMaxRank or
				not modelFileValidScalarType(entry.dtype)) {
				return modelFileCorrupt("invalid tensor metadata");
			}
			oa::U64 entryEnd = 0;
			if (not modelFileCheckedAdd(entry.blobOffset, entry.numBytes, entryEnd) or
				entryEnd > blobSize) {
				return modelFileCorrupt("tensor payload range is outside its section");
			}
			if (not names.insert(oa::String(entry.name)).second) {
				return modelFileCorrupt("duplicate tensor name");
			}
			auto expected = modelFileExpectedTensorBytes(entry, inType == ModelFileSection::Weights,
														 fh.version);
			if (not expected.isOk() or expected.getValue() != entry.numBytes) {
				return modelFileCorrupt("tensor byte count does not match shape/encoding");
			}
			if (entry.numBytes != 0)
				tensorRanges.emplaceBack(entry.blobOffset, entryEnd);
		}
		oa::sort(tensorRanges.begin(), tensorRanges.end(),
			[](const auto& inLeft, const auto& inRight) {
				return inLeft.first < inRight.first;
			});
		for (oa::Usize i = 1; i < tensorRanges.size(); ++i) {
			if (tensorRanges[i].first < tensorRanges[i - 1].second) {
				return modelFileCorrupt("overlapping tensor payload ranges");
			}
		}
		outBlob.resize(static_cast<oa::Usize>(blobSize));
		if (blobSize > 0 and
			not modelFileReadExact(file.value, sh->offset + indexBytes, outBlob.data(), blobSize)) {
			return modelFileCorrupt("truncated tensor payload");
		}
		return oa::Status::ok();
	};
	OA_RETURN_IF_ERROR(
		parseTensorSection(ModelFileSection::Weights, model.weightIndex, model.weightBlob));
	OA_RETURN_IF_ERROR(
		parseTensorSection(ModelFileSection::State, model.stateIndex, model.stateBlob));

	if (const auto* sh = findSection(ModelFileSection::Optimizer)) {
		model.optimizerPresent = true;
		if (sh->size < sizeof(ModelOptimizerState) or
			not modelFileReadExact(file.value, sh->offset, &model.optimizer,
								   sizeof(ModelOptimizerState)) or
			not modelFileHasTerminator(model.optimizer.type, sizeof(model.optimizer.type))) {
			return modelFileCorrupt("invalid optimizer header");
		}
		if (not modelFileHasKnownOptimizer(model.optimizer)) {
			return modelFileCorrupt("unknown optimizer type");
		}
		const oa::U64 adamNum = model.optimizer.numParams;
		oa::U64 adamBytes = 0;
		oa::U64 expectedSize = sizeof(ModelOptimizerState);
		if (not modelFileCheckedMul(adamNum, sizeof(oa::F32), adamBytes) or
			not modelFileCheckedAdd(expectedSize, adamBytes, expectedSize)) {
			return modelFileCorrupt("optimizer size overflow");
		}
		if (not modelFileIsMuonOnly(model.optimizer) and
			not modelFileCheckedAdd(expectedSize, adamBytes, expectedSize)) {
			return modelFileCorrupt("optimizer size overflow");
		}
		if (expectedSize != sh->size) {
			return modelFileCorrupt("optimizer payload size mismatch");
		}
		oa::U64 offset = sizeof(ModelOptimizerState);
		model.adamM.resize(static_cast<oa::Usize>(adamNum));
		if (adamBytes > 0 and
			not modelFileReadExact(file.value, sh->offset + offset, model.adamM.data(), adamBytes)) {
			return modelFileCorrupt("truncated optimizer first moment");
		}
		offset += adamBytes;
		if (not modelFileIsMuonOnly(model.optimizer)) {
			model.adamV.resize(static_cast<oa::Usize>(adamNum));
			if (adamBytes > 0 and
				not modelFileReadExact(file.value, sh->offset + offset, model.adamV.data(), adamBytes)) {
				return modelFileCorrupt("truncated optimizer second moment");
			}
			offset += adamBytes;
		}
	}

	const auto* progressSection = findSection(ModelFileSection::Progress);
	if (progressSection->size != sizeof(ModelTrainingProgress) or
		not modelFileReadExact(file.value, progressSection->offset, &model.progress,
							   sizeof(ModelTrainingProgress)) or
		not modelFileHasTerminator(model.progress.metricName, sizeof(model.progress.metricName))) {
		return modelFileCorrupt("invalid progress section");
	}

	OaLogInfo(oa::LogComponent::Ml, "[oam] Loaded {} | arch={} | {} weights | optimizer={}",
			  inPath.cStr(), model.config.architecture, model.weightIndex.size(),
			  model.hasOptimizer() ? "yes" : "no");
	return model;
}

// Dump

void dumpModelFile(const oa::String& inPath) {
	{
		auto verified = ModelFile::load(inPath);
		if (not verified.isOk()) {
			OaLogError(oa::LogComponent::Ml, "[oam] Refusing to inspect {}: {}", inPath.cStr(),
					   verified.getStatus().toString().cStr());
			return;
		}
	}
	ModelFileHandle file(::fopen(inPath.cStr(), "rb"));
	if (not file.isOpen()) {
		OaLogError(oa::LogComponent::Ml, "[oam] Cannot open: {}", inPath.cStr());
		return;
	}

	ModelFileHeader fh;
	if (not modelFileReadExact(file.value, 0, &fh, sizeof(fh)) or
		fh.magic != kModelFileMagic) {
		OaLogError(oa::LogComponent::Ml, "[oam] Not an .oam file");
		return;
	}

	OaLogInfo(oa::LogComponent::Ml, "=== {} ===", inPath.cStr());
	OaLogInfo(oa::LogComponent::Ml, "  Sections: {}  size: {:.1f} MB", fh.numSections,
			  fh.totalSize / 1e6);

	oa::Vector<ModelFileSectionHeader> sections(fh.numSections);
	if (not modelFileReadExact(file.value, sizeof(ModelFileHeader), sections.data(),
							   sections.size() * sizeof(ModelFileSectionHeader))) {
		OaLogError(oa::LogComponent::Ml, "[oam] Cannot read section table");
		return;
	}

	for (oa::U32 i = 0; i < fh.numSections; ++i) {
		const auto& sh = sections[i];
		auto type = static_cast<ModelFileSection>(sh.type);
		OaLogInfo(oa::LogComponent::Ml, "  [{}] {}  offset={}  {:.2f} MB", i,
				  modelFileSectionName(type), sh.offset, sh.size / 1e6);

		if (type == ModelFileSection::Config) {
			ModelFileConfig cfg;
			if (not modelFileReadExact(file.value, sh.offset, &cfg,
					oa::min(sh.size, static_cast<oa::U64>(sizeof(ModelFileConfig))))) continue;
			OaLogInfo(oa::LogComponent::Ml, "       arch={} dModel={} nLayers={} dVocab={}",
					  cfg.architecture, cfg.dModel, cfg.nLayers, cfg.dVocab);
		}

		if (type == ModelFileSection::Weights || type == ModelFileSection::State) {
			oa::U32 count = 0, reserved = 0;
			if (not modelFileReadExact(file.value, sh.offset, &count, sizeof(oa::U32)) or
				not modelFileReadExact(file.value, sh.offset + sizeof(oa::U32), &reserved,
								   sizeof(oa::U32))) continue;
			for (oa::U32 j = 0; j < count; ++j) {
				ModelTensorEntry e;
				const oa::U64 entryOffset = sh.offset + sizeof(oa::U32) * 2
					+ static_cast<oa::U64>(j) * sizeof(ModelTensorEntry);
				if (not modelFileReadExact(file.value, entryOffset, &e, sizeof(e))) break;
				OaLogInfo(oa::LogComponent::Ml, "       {}  {}/{}  {:.3f} MB", e.name,
						  modelFileTensorEncodingName(e.encoding),
						  oa::scalarTypeName(e.dtype), e.numBytes / 1e6);
			}
		}
	}
}

} // namespace oa
