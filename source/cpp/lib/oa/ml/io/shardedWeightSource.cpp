#include "shardedWeightSource.h"

#include <oa/core/filesystem.h>
#include <oa/core/hostText.h>
#include <oa/core/std/limits.h>
#include <oa/core/std/pair.h>

#include <yaml-cpp/yaml.h>

namespace oa {

oa::Status ShardedWeightSource::open(const oa::Path& inIndexPath) {
	path_ = inIndexPath;
	sources_.clear();
	infos_.clear();
	locations_.clear();
	metadata_.clear();
	sourceBytes_ = 0;

	YAML::Node root;
	try {
		root = YAML::LoadFile(oa::hostText::copy(inIndexPath.string()));
	} catch (const std::exception& error) {
		return oa::Status::error(oa::StatusCode::FileCorrupt,
			oa::String("Cannot parse weight index: ") + error.what());
	}
	const auto weightMap = root["weight_map"];
	if (!weightMap || !weightMap.IsMap() || weightMap.size() == 0) {
		return oa::Status::error(oa::StatusCode::FileCorrupt, "weight index has no non-empty weight_map");
	}
	const auto metadata = root["metadata"];
	if (metadata && metadata.IsMap()) {
		for (const auto& item : metadata) {
			if (item.first.IsScalar() && item.second.IsScalar()) {
				metadata_.insert({
					oa::hostText::copy(item.first.Scalar()),
					oa::hostText::copy(item.second.Scalar())
				});
			}
		}
	}

	oa::HashMap<oa::String, oa::Usize> shardIndices;
	oa::HashSet<oa::String> indexedNames;
	oa::Vector<oa::Pair<oa::String, oa::String>> indexedWeights;
	indexedWeights.reserve(weightMap.size());
	for (const auto& item : weightMap) {
		if (!item.first.IsScalar() || !item.second.IsScalar()) {
			return oa::Status::error(oa::StatusCode::FileCorrupt, "weight_map names and shards must be strings");
		}
		const oa::String name = oa::hostText::copy(item.first.Scalar());
		const oa::String shard = oa::hostText::copy(item.second.Scalar());
		if (name.empty() || shard.empty()) {
			return oa::Status::error(oa::StatusCode::FileCorrupt, "weight_map contains an empty name or shard");
		}
		if (!indexedNames.insert(name).second) {
			return oa::Status::error(oa::StatusCode::FileCorrupt,
				oa::String("Duplicate global weight name in index: ") + name);
		}
		indexedWeights.pushBack({name, shard});
		if (!shardIndices.contains(shard)) {
			const oa::Path shardPath = inIndexPath.parentPath() / oa::Path(shard);
			auto sourceResult = openWeightSource(shardPath, oa::WeightFormat::SafeTensors);
			if (sourceResult.isError()) return sourceResult.getStatus();
			const oa::Usize index = sources_.size();
			const oa::U64 shardBytes = sourceResult.getValue()->sourceBytes();
			if (sourceBytes_ > oa::Limits<oa::U64>::max() - shardBytes) {
				return oa::Status::error(oa::StatusCode::OutOfRange, "weight package byte count overflow");
			}
			sourceBytes_ += shardBytes;
			sources_.pushBack(oa::move(sourceResult.getValue()));
			shardIndices.insert({shard, index});
		}
	}
	auto indexSize = oa::Filesystem::getFileSize(inIndexPath);
	if (indexSize.isOk()) {
		const oa::U64 indexBytes = static_cast<oa::U64>(indexSize.getValue());
		if (sourceBytes_ > oa::Limits<oa::U64>::max() - indexBytes) {
			return oa::Status::error(oa::StatusCode::OutOfRange, "weight package byte count overflow");
		}
		sourceBytes_ += indexBytes;
	}

	for (const auto& [name, shard] : indexedWeights) {
		const oa::Usize sourceIndex = shardIndices.at(shard);
		const auto* info = sources_[sourceIndex]->find(name);
		if (!info) {
			return oa::Status::error(oa::StatusCode::FileCorrupt,
				oa::String("Indexed weight is missing from shard: ") + name);
		}
		const oa::Usize infoIndex = infos_.size();
		infos_.pushBack(*info);
		locations_.insert({name, Location{sourceIndex, infoIndex}});
	}

	for (oa::Usize sourceIndex = 0; sourceIndex < sources_.size(); ++sourceIndex) {
		for (const auto& info : sources_[sourceIndex]->list()) {
			auto location = locations_.find(info.name);
			if (location == locations_.end() || location->second.sourceIndex != sourceIndex) {
				return oa::Status::error(oa::StatusCode::FileCorrupt,
					oa::String("Shard contains an unindexed or misindexed weight: ") + info.name);
			}
		}
	}
	return oa::Status::ok();
}

const oa::WeightInfo* ShardedWeightSource::find(oa::StringView inName) const {
	auto location = locations_.find(oa::String(inName));
	return location == locations_.end() ? nullptr : &infos_[location->second.infoIndex];
}

oa::Result<oa::Span<const oa::U8>> ShardedWeightSource::bytes(oa::StringView inName) const {
	auto location = locations_.find(oa::String(inName));
	if (location == locations_.end()) return oa::Status::notFound(oa::String("weight not found: ") + inName);
	return sources_[location->second.sourceIndex]->bytes(inName);
}

oa::Status ShardedWeightSource::read(
	oa::StringView inName, oa::Span<oa::U8> outData, oa::ScalarType inTargetDtype) const {
	auto location = locations_.find(oa::String(inName));
	if (location == locations_.end()) return oa::Status::notFound(oa::String("weight not found: ") + inName);
	return sources_[location->second.sourceIndex]->read(inName, outData, inTargetDtype);
}

} // namespace oa
