#pragma once

#include <Oa/Core/MappedFile.h>
#include <Oa/Ml/TransferWeights.h>

// Private backend for the SafeTensors container format. Do not expose this type
// through model APIs; use OaOpenWeightSource and OaTransferWeights.
class OaSafeTensorsWeightSource final : public OaWeightSource {
public:
	OaStatus Open(const OaPath& InPath);

	[[nodiscard]] OaWeightFormat Format() const noexcept override {
		return OaWeightFormat::SafeTensors;
	}
	[[nodiscard]] const OaPath& Path() const noexcept override { return Path_; }
	[[nodiscard]] OaVec<OaWeightInfo> List() const override;
	[[nodiscard]] const OaWeightInfo* Find(OaStringView InName) const override;
	[[nodiscard]] OaResult<OaSpan<const OaU8>> Bytes(OaStringView InName) const override;
	[[nodiscard]] OaHashMap<OaString, OaString> Metadata() const override { return Metadata_; }
	[[nodiscard]] OaU64 SourceBytes() const noexcept override {
		return static_cast<OaU64>(File_.Size());
	}
	OaStatus Read(OaStringView InName, OaSpan<OaU8> OutData,
		OaScalarType InTargetDtype) const override;

	[[nodiscard]] OaU64 HeaderSize() const noexcept { return HeaderLen_; }

private:
	struct Entry {
		OaWeightInfo Info;
		OaU64 DataOffset = 0;
	};

	OaStatus ParseHeader(OaSpan<const OaU8> InHeaderData);
	OaStatus ValidateEntries();
	OaResult<OaScalarType> ParseDtype(OaStringView InDtype) const;

	OaPath Path_;
	bool IsOpen_ = false;
	OaU64 HeaderLen_ = 0;
	OaU64 DataStart_ = 0;
	OaMappedFile File_;
	OaHashMap<OaString, Entry> Entries_;
	OaVec<OaString> EntryOrder_;
	OaHashMap<OaString, OaString> Metadata_;
};
