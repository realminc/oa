#pragma once

#include <oa/core/mappedFile.h>
#include <oa/ml/transferWeights.h>

// Private backend for the SafeTensors container format. Do not expose this type
// through model APIs; use oa::openWeightSource and oa::transferWeights.
namespace oa {

class SafeTensorsWeightSource final : public WeightSource {
public:
	Status open(const Path& inPath);

	[[nodiscard]] WeightFormat format() const noexcept override {
		return WeightFormat::SafeTensors;
	}
	[[nodiscard]] const Path& path() const noexcept override { return path_; }
	[[nodiscard]] Vector<WeightInfo> list() const override;
	[[nodiscard]] const WeightInfo* find(StringView inName) const override;
	[[nodiscard]] Result<Span<const U8>> bytes(StringView inName) const override;
	[[nodiscard]] HashMap<String, String> metadata() const override { return metadata_; }
	[[nodiscard]] U64 sourceBytes() const noexcept override {
		return static_cast<U64>(file_.size());
	}
	Status read(StringView inName, Span<U8> outData,
		ScalarType inTargetDtype) const override;

	[[nodiscard]] U64 headerSize() const noexcept { return headerLen_; }

private:
	struct Entry {
		WeightInfo info;
		U64 dataOffset = 0;
	};

	Status parseHeader(Span<const U8> inHeaderData);
	Status validateEntries();
	Result<ScalarType> parseDtype(StringView inDtype) const;

	Path path_;
	bool isOpen_ = false;
	U64 headerLen_ = 0;
	U64 dataStart_ = 0;
	MappedFile file_;
	HashMap<String, Entry> entries_;
	Vector<String> entryOrder_;
	HashMap<String, String> metadata_;
};

} // namespace oa
