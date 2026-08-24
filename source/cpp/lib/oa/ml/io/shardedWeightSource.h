#pragma once

#include <oa/ml/transferWeights.h>

// Private aggregate source for standard external weight-index packages. Each
// shard is opened through a checked container backend and exposed as one global
// immutable name space.
namespace oa {

class ShardedWeightSource final : public WeightSource {
public:
	Status open(const Path& inIndexPath);

	[[nodiscard]] WeightFormat format() const noexcept override { return WeightFormat::SafeTensors; }
	[[nodiscard]] const Path& path() const noexcept override { return path_; }
	[[nodiscard]] Vec<WeightInfo> list() const override { return infos_; }
	[[nodiscard]] const WeightInfo* find(StringView inName) const override;
	[[nodiscard]] Result<Span<const U8>> bytes(StringView inName) const override;
	[[nodiscard]] HashMap<String, String> metadata() const override { return metadata_; }
	[[nodiscard]] U64 sourceBytes() const noexcept override { return sourceBytes_; }
	Status read(StringView inName, Span<U8> outData,
		ScalarType inTargetDtype) const override;

private:
	struct Location {
		Usize sourceIndex = 0;
		Usize infoIndex = 0;
	};

	Path path_;
	Vec<UniquePtr<WeightSource>> sources_;
	Vec<WeightInfo> infos_;
	HashMap<String, Location> locations_;
	HashMap<String, String> metadata_;
	U64 sourceBytes_ = 0;
};

} // namespace oa
