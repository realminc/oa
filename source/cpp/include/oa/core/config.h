// YAML helpers and shared checkpoint/logging config. CLI: <oa/core/cli.h>.

#pragma once

#include <oa/core/types.h>
#include <oa/core/filesystem.h>
#include <oa/core/paths.h>
#include <oa/core/yaml.h>

namespace oa {

class CheckpointConfig {
public:
	oa::String dir = "checkpoints/";
	oa::String name = "model";
	oa::String env = "dev";
	oa::Bool saveBest = true;
	oa::Bool saveLast = true;
};

class LogConfig {
public:
	oa::String dir = oa::Paths::var("log").string();
	oa::String level = "info";
	oa::Bool console = true;
	oa::Bool file = false;
	oa::Bool metrics = true;
};

inline void loadCheckpointYaml(const oa::Yaml::Node& inYaml, CheckpointConfig& outCfg) {
	if (auto c = inYaml["checkpoint"]) {
		outCfg.dir = oa::Yaml::get<oa::String>(c, "dir", outCfg.dir);
		outCfg.name = oa::Yaml::get<oa::String>(c, "name", outCfg.name);
		outCfg.env = oa::Yaml::get<oa::String>(c, "env", outCfg.env);
		outCfg.saveBest = oa::Yaml::get<bool>(c, "save_best", outCfg.saveBest);
		outCfg.saveLast = oa::Yaml::get<bool>(c, "save_last", outCfg.saveLast);
	}
}

inline void loadLogYaml(const oa::Yaml::Node& inYaml, LogConfig& outCfg) {
	if (auto l = inYaml["logging"]) {
		outCfg.dir = oa::Yaml::get<oa::String>(l, "dir", outCfg.dir);
		outCfg.level = oa::Yaml::get<oa::String>(l, "level", outCfg.level);
		outCfg.console = oa::Yaml::get<bool>(l, "console", outCfg.console);
		outCfg.file = oa::Yaml::get<bool>(l, "file", outCfg.file);
		outCfg.metrics = oa::Yaml::get<bool>(l, "metrics", outCfg.metrics);
	}
}

} // namespace oa
