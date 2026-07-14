#pragma once

#include <Oa/Ml/TransferWeights.h>

// Private aggregate source for standard external weight-index packages. Each
// shard is opened through a checked container backend and exposed as one global
// immutable name space.
class OaShardedWeightSource final : public OaWeightSource {
public:
	OaStatus Open(const OaPath& InIndexPath);

	[[nodiscard]] OaWeightFormat Format() const noexcept override { return OaWeightFormat::SafeTensors; }
	[[nodiscard]] const OaPath& Path() const noexcept override { return Path_; }
	[[nodiscard]] OaVec<OaWeightInfo> List() const override { return Infos_; }
	[[nodiscard]] const OaWeightInfo* Find(OaStringView InName) const override;
	[[nodiscard]] OaResult<OaSpan<const OaU8>> Bytes(OaStringView InName) const override;
	[[nodiscard]] OaHashMap<OaString, OaString> Metadata() const override { return Metadata_; }
	[[nodiscard]] OaU64 SourceBytes() const noexcept override { return SourceBytes_; }
	OaStatus Read(OaStringView InName, OaSpan<OaU8> OutData,
		OaScalarType InTargetDtype) const override;

private:
	struct Location {
		OaUsize SourceIndex = 0;
		OaUsize InfoIndex = 0;
	};

	OaPath Path_;
	OaVec<OaUniquePtr<OaWeightSource>> Sources_;
	OaVec<OaWeightInfo> Infos_;
	OaHashMap<OaString, Location> Locations_;
	OaHashMap<OaString, OaString> Metadata_;
	OaU64 SourceBytes_ = 0;
};
