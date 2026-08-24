// oa::Cli<T> — 3-Way Precedence CLI
//
// Precedence: struct defaults < YAML file < explicit CLI args
//
// Flow:
//   1. cfg_ starts with hardcoded defaults (struct member initializers)
//   2. Extract --config path from argv (before CLI11 touches anything)
//   3. load YAML -> overwrites defaults where specified
//   4. CLI11 parse -> only overwrites values user EXPLICITLY typed
//
// Subclass pattern:
//   struct TrainConfig { oa::F32 lr = 3e-4f; oa::String data; };
//   class TrainCli : public oa::Cli<TrainConfig> {
//       TrainCli() : oa::Cli("train", "Train a model") {
//           addOption("--lr", cfg_.lr, "Learning rate");
//           addOption("--data,-d", cfg_.data, "Data file");
//       }
//       void loadYaml(const oa::Yaml::Node& yaml) override { ... }
//   };
//
// dependencies: CLI11, yaml-cpp, fmt (all in vcpkg)

#pragma once

#include <oa/core/yaml.h>

// CLI11 and fmt are optional dependencies
// To enable: install via vcpkg and define OA_HAS_CLI11
#ifdef OA_HAS_CLI11
#include <CLI/CLI.hpp>
#include <fmt/format.h>

namespace oa {

template<typename TConfig>
class Cli {
public:

	// Data, class members.
	CLI::App app_;
	TConfig cfg_;
	oa::String configPath_;
	oa::I32 verbose_ = 0;

	// Constructors.
	Cli(const oa::String& inName, const oa::String& inDescription)
		: app_(inDescription.stdStr(), inName.stdStr())
	{
		app_.add_option("-c,--config", configPath_, "YAML config file");
		app_.add_option("-v,--verbose", verbose_, "verbose level (0-3)")->default_val(0);
		app_.set_help_all_flag("--help-all", "show all options");
	}

	// Destructors.
	virtual ~Cli() = default;

	// Methods.
	bool parse(int inArgc, char** inArgv) {
		// step 2: Extract --config / -c from argv BEFORE CLI11 parse
		for (int i = 1; i < inArgc; ++i) {
			oa::String arg(inArgv[i]);
			if ((arg == "--config" || arg == "-c") && i + 1 < inArgc) {
				configPath_ = inArgv[i + 1];
				break;
			}
			if (arg.substr(0, 9) == "--config=") {
				configPath_ = arg.substr(9);
				break;
			}
			if (arg.substr(0, 3) == "-c=") {
				configPath_ = arg.substr(3);
				break;
			}
		}

		// step 3: load YAML -> overwrites defaults with YAML values
		if (!configPath_.empty()) {
			try {
				oa::Yaml::Node yaml = oa::Yaml::loadFile(configPath_);
				loadYaml(yaml);
			} catch (const oa::Yaml::Exception& e) {
				fmt::print(stderr, "[OA CONFIG] YAML load failed: {} (using defaults)\n", e.what());
			}
		}

		// step 4: CLI11 parse -> ONLY modifies cfg_ fields where user
		//         explicitly provided a CLI argument
		try {
			app_.parse(inArgc, inArgv);
			applyCliOverrides();
			return true;
		} catch (const CLI::ParseError& e) {
			app_.exit(e);
			return false;
		}
	}

	[[nodiscard]] const TConfig& getConfig() const { return cfg_; }
	[[nodiscard]] TConfig& getConfig() { return cfg_; }
	[[nodiscard]] const oa::String& getConfigPath() const { return configPath_; }
	[[nodiscard]] oa::I32 getVerbose() const { return verbose_; }

	// Subcommands
	CLI::App* addSubcommand(const oa::String& inName, const oa::String& inDesc) {
		return app_.add_subcommand(inName.stdStr(), inDesc.stdStr());
	}

	[[nodiscard]] bool gotSubcommand(const oa::String& inName) const {
		return app_.got_subcommand(inName.stdStr());
	}

	[[nodiscard]] oa::String getSubcommand() const {
		for (auto* sub : app_.get_subcommands()) {
			if (sub->parsed()) {
				return oa::String(sub->get_name());
			}
		}
		return "";
	}

	void requireSubcommand(oa::I32 inMin = 0, oa::I32 inMax = 0) {
		app_.require_subcommand(inMin, inMax);
	}

	void setEpilog(const oa::String& inText) {
		app_.footer(inText.stdStr());
	}

	// Allow fallthrough so global options work with subcommands
	void setFallthrough(bool inEnable = true) {
		app_.fallthrough(inEnable);
	}

protected:
	virtual void loadYaml(const oa::Yaml::Node& inYaml) { (void)inYaml; }
	virtual void applyCliOverrides() {}

	template<typename T>
	CLI::Option* addOption(const oa::String& inFlag, T& inTarget, const oa::String& inDesc) {
		auto* opt = app_.add_option(inFlag.stdStr(), inTarget, inDesc.stdStr());
		storeOptionPtr(inFlag, opt);
		return opt;
	}

	CLI::Option* addFlag(const oa::String& inFlag, bool& inTarget, const oa::String& inDesc) {
		auto* opt = app_.add_flag(inFlag.stdStr(), inTarget, inDesc.stdStr());
		storeOptionPtr(inFlag, opt);
		return opt;
	}

	CLI::Option* addMultiOption(const oa::String& inFlag, oa::Vec<oa::String>& inTarget, const oa::String& inDesc) {
		auto* opt = app_.add_option(inFlag.stdStr(), inTarget, inDesc.stdStr());
		storeOptionPtr(inFlag, opt);
		return opt;
	}

	template<typename T>
	CLI::Option* addPositional(const oa::String& inName, T& inTarget, const oa::String& inDesc, bool inRequired = false) {
		auto* opt = app_.add_option(inName.stdStr(), inTarget, inDesc.stdStr());
		if (inRequired) {
			opt->required();
		}
		return opt;
	}

	[[nodiscard]] bool wasSet(const oa::String& inFlag) const {
		auto it = options_.find(inFlag);
		if (it != options_.end()) {
			return it->second->count() > 0;
		}
		return false;
	}

	void storeOptionPtr(const oa::String& inFlag, CLI::Option* inOpt) {
		auto it = options_.find(inFlag);
		if (it != options_.end()) {
			it->second = inOpt;
		} else {
			options_.emplace(inFlag, inOpt);
		}
	}

private:
	oa::HashMap<oa::String, CLI::Option*> options_;
};

} // namespace oa

#endif // OA_HAS_CLI11
