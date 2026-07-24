// YAML helpers + shared checkpoint/logging config. CLI: Oa/Core/Cli.h.

#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/Filesystem.h>
#include <Oa/Core/Paths.h>
#include <Oa/Core/Yaml.h>

class OaCheckpointConfig {
public:
	OaString Dir = "checkpoints/";
	OaString Name = "model";
	OaString Env = "dev";
	OaBool SaveBest = true;
	OaBool SaveLast = true;
};

class OaLogConfig {
public:
	OaString Dir = OaPaths::Var("log").String();
	OaString Level = "info";
	OaBool Console = true;
	OaBool File = false;
	OaBool Metrics = true;
};

inline void OaLoadCheckpointYaml(const OaYaml::Node& InYaml, OaCheckpointConfig& OutCfg) {
	if (auto c = InYaml["checkpoint"]) {
		OutCfg.Dir = OaYaml::Get<OaString>(c, "dir", OutCfg.Dir);
		OutCfg.Name = OaYaml::Get<OaString>(c, "name", OutCfg.Name);
		OutCfg.Env = OaYaml::Get<OaString>(c, "env", OutCfg.Env);
		OutCfg.SaveBest = OaYaml::Get<bool>(c, "save_best", OutCfg.SaveBest);
		OutCfg.SaveLast = OaYaml::Get<bool>(c, "save_last", OutCfg.SaveLast);
	}
}

inline void OaLoadLogYaml(const OaYaml::Node& InYaml, OaLogConfig& OutCfg) {
	if (auto l = InYaml["logging"]) {
		OutCfg.Dir = OaYaml::Get<OaString>(l, "dir", OutCfg.Dir);
		OutCfg.Level = OaYaml::Get<OaString>(l, "level", OutCfg.Level);
		OutCfg.Console = OaYaml::Get<bool>(l, "console", OutCfg.Console);
		OutCfg.File = OaYaml::Get<bool>(l, "file", OutCfg.File);
		OutCfg.Metrics = OaYaml::Get<bool>(l, "metrics", OutCfg.Metrics);
	}
}
