#include "ShardedWeightSource.h"

#include <Oa/Core/FileIo.h>

#include <yaml-cpp/yaml.h>

#include <limits>

OaStatus OaShardedWeightSource::Open(const OaPath& InIndexPath) {
	Path_ = InIndexPath;
	Sources_.Clear();
	Infos_.Clear();
	Locations_.Clear();
	Metadata_.Clear();
	SourceBytes_ = 0;

	YAML::Node root;
	try {
		root = YAML::LoadFile(InIndexPath.String().StdStr());
	} catch (const std::exception& error) {
		return OaStatus::Error(OaStatusCode::FileCorrupt,
			OaString("Cannot parse weight index: ") + error.what());
	}
	const auto weightMap = root["weight_map"];
	if (!weightMap || !weightMap.IsMap() || weightMap.size() == 0) {
		return OaStatus::Error(OaStatusCode::FileCorrupt, "Weight index has no non-empty weight_map");
	}
	const auto metadata = root["metadata"];
	if (metadata && metadata.IsMap()) {
		for (const auto& item : metadata) {
			if (item.first.IsScalar() && item.second.IsScalar()) {
				Metadata_.Insert({OaString(item.first.Scalar()), OaString(item.second.Scalar())});
			}
		}
	}

	OaHashMap<OaString, OaUsize> shardIndices;
	OaHashSet<OaString> indexedNames;
	OaVec<std::pair<OaString, OaString>> indexedWeights;
	indexedWeights.Reserve(weightMap.size());
	for (const auto& item : weightMap) {
		if (!item.first.IsScalar() || !item.second.IsScalar()) {
			return OaStatus::Error(OaStatusCode::FileCorrupt, "weight_map names and shards must be strings");
		}
		const OaString name(item.first.Scalar());
		const OaString shard(item.second.Scalar());
		if (name.empty() || shard.empty()) {
			return OaStatus::Error(OaStatusCode::FileCorrupt, "weight_map contains an empty name or shard");
		}
		if (!indexedNames.Insert(name).second) {
			return OaStatus::Error(OaStatusCode::FileCorrupt,
				OaString("Duplicate global weight name in index: ") + name);
		}
		indexedWeights.PushBack({name, shard});
		if (!shardIndices.Contains(shard)) {
			const OaPath shardPath = InIndexPath.ParentPath() / OaPath(shard);
			auto sourceResult = OaOpenWeightSource(shardPath, OaWeightFormat::SafeTensors);
			if (sourceResult.IsError()) return sourceResult.GetStatus();
			const OaUsize index = Sources_.Size();
			const OaU64 shardBytes = sourceResult.GetValue()->SourceBytes();
			if (SourceBytes_ > std::numeric_limits<OaU64>::max() - shardBytes) {
				return OaStatus::Error(OaStatusCode::OutOfRange, "Weight package byte count overflow");
			}
			SourceBytes_ += shardBytes;
			Sources_.PushBack(OaStdMove(sourceResult.GetValue()));
			shardIndices.Insert({shard, index});
		}
	}
	auto indexSize = OaFileIo::GetFileSize(InIndexPath);
	if (indexSize.IsOk()) {
		const OaU64 indexBytes = static_cast<OaU64>(indexSize.GetValue());
		if (SourceBytes_ > std::numeric_limits<OaU64>::max() - indexBytes) {
			return OaStatus::Error(OaStatusCode::OutOfRange, "Weight package byte count overflow");
		}
		SourceBytes_ += indexBytes;
	}

	for (const auto& [name, shard] : indexedWeights) {
		const OaUsize sourceIndex = shardIndices.At(shard);
		const auto* info = Sources_[sourceIndex]->Find(name);
		if (!info) {
			return OaStatus::Error(OaStatusCode::FileCorrupt,
				OaString("Indexed weight is missing from shard: ") + name);
		}
		const OaUsize infoIndex = Infos_.Size();
		Infos_.PushBack(*info);
		Locations_.Insert({name, Location{sourceIndex, infoIndex}});
	}

	for (OaUsize sourceIndex = 0; sourceIndex < Sources_.Size(); ++sourceIndex) {
		for (const auto& info : Sources_[sourceIndex]->List()) {
			auto location = Locations_.Find(info.Name);
			if (location == Locations_.End() || location->second.SourceIndex != sourceIndex) {
				return OaStatus::Error(OaStatusCode::FileCorrupt,
					OaString("Shard contains an unindexed or misindexed weight: ") + info.Name);
			}
		}
	}
	return OaStatus::Ok();
}

const OaWeightInfo* OaShardedWeightSource::Find(OaStringView InName) const {
	auto location = Locations_.Find(OaString(InName));
	return location == Locations_.End() ? nullptr : &Infos_[location->second.InfoIndex];
}

OaResult<OaSpan<const OaU8>> OaShardedWeightSource::Bytes(OaStringView InName) const {
	auto location = Locations_.Find(OaString(InName));
	if (location == Locations_.End()) return OaStatus::NotFound(OaString("Weight not found: ") + InName);
	return Sources_[location->second.SourceIndex]->Bytes(InName);
}

OaStatus OaShardedWeightSource::Read(
	OaStringView InName, OaSpan<OaU8> OutData, OaScalarType InTargetDtype) const {
	auto location = Locations_.Find(OaString(InName));
	if (location == Locations_.End()) return OaStatus::NotFound(OaString("Weight not found: ") + InName);
	return Sources_[location->second.SourceIndex]->Read(InName, OutData, InTargetDtype);
}
