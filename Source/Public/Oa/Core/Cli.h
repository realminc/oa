// OaCli<T> — 3-Way Precedence CLI
//
// Precedence: struct defaults < YAML file < explicit CLI args
//
// Flow:
//   1. Cfg_ starts with hardcoded defaults (struct member initializers)
//   2. Extract --config path from argv (before CLI11 touches anything)
//   3. Load YAML -> overwrites defaults where specified
//   4. CLI11 parse -> only overwrites values user EXPLICITLY typed
//
// Subclass pattern:
//   struct TrainConfig { OaF32 Lr = 3e-4f; OaString Data; };
//   class TrainCli : public OaCli<TrainConfig> {
//       TrainCli() : OaCli("train", "Train a model") {
//           AddOption("--lr", Cfg_.Lr, "Learning rate");
//           AddOption("--data,-d", Cfg_.Data, "Data file");
//       }
//       void LoadYaml(const OaYaml::Node& yaml) override { ... }
//   };
//
// Dependencies: CLI11, yaml-cpp, fmt (all in vcpkg)

#pragma once

#include <Oa/Core/Yaml.h>

// CLI11 and fmt are optional dependencies
// To enable: install via vcpkg and define OA_HAS_CLI11
#ifdef OA_HAS_CLI11
#include <CLI/CLI.hpp>
#include <fmt/format.h>

template<typename TConfig>
class OaCli {
public:

	// Data, class members.
	CLI::App App_;
	TConfig Cfg_;
	OaString ConfigPath_;
	OaI32 Verbose_ = 0;

	// Constructors.
	OaCli(const OaString& InName, const OaString& InDescription)
		: App_(InDescription.StdStr(), InName.StdStr())
	{
		App_.add_option("-c,--config", ConfigPath_, "YAML config file");
		App_.add_option("-v,--verbose", Verbose_, "Verbose level (0-3)")->default_val(0);
		App_.set_help_all_flag("--help-all", "Show all options");
	}

	// Destructors.
	virtual ~OaCli() = default;

	// Methods.
	bool Parse(int InArgc, char** InArgv) {
		// Step 2: Extract --config / -c from argv BEFORE CLI11 parse
		for (int i = 1; i < InArgc; ++i) {
			OaString arg(InArgv[i]);
			if ((arg == "--config" || arg == "-c") && i + 1 < InArgc) {
				ConfigPath_ = InArgv[i + 1];
				break;
			}
			if (arg.substr(0, 9) == "--config=") {
				ConfigPath_ = arg.substr(9);
				break;
			}
			if (arg.substr(0, 3) == "-c=") {
				ConfigPath_ = arg.substr(3);
				break;
			}
		}

		// Step 3: Load YAML -> overwrites defaults with YAML values
		if (!ConfigPath_.empty()) {
			try {
				OaYaml::Node yaml = OaYaml::LoadFile(ConfigPath_);
				LoadYaml(yaml);
			} catch (const OaYaml::Exception& e) {
				fmt::print(stderr, "[OA CONFIG] YAML load failed: {} (using defaults)\n", e.what());
			}
		}

		// Step 4: CLI11 parse -> ONLY modifies Cfg_ fields where user
		//         explicitly provided a CLI argument
		try {
			App_.parse(InArgc, InArgv);
			ApplyCliOverrides();
			return true;
		} catch (const CLI::ParseError& e) {
			App_.exit(e);
			return false;
		}
	}

	[[nodiscard]] const TConfig& GetConfig() const { return Cfg_; }
	[[nodiscard]] TConfig& GetConfig() { return Cfg_; }
	[[nodiscard]] const OaString& GetConfigPath() const { return ConfigPath_; }
	[[nodiscard]] OaI32 GetVerbose() const { return Verbose_; }

	// Subcommands
	CLI::App* AddSubcommand(const OaString& InName, const OaString& InDesc) {
		return App_.add_subcommand(InName.StdStr(), InDesc.StdStr());
	}

	[[nodiscard]] bool GotSubcommand(const OaString& InName) const {
		return App_.got_subcommand(InName.StdStr());
	}

	[[nodiscard]] OaString GetSubcommand() const {
		for (auto* sub : App_.get_subcommands()) {
			if (sub->parsed()) {
				return OaString(sub->get_name());
			}
		}
		return "";
	}

	void RequireSubcommand(OaI32 InMin = 0, OaI32 InMax = 0) {
		App_.require_subcommand(InMin, InMax);
	}

	void SetEpilog(const OaString& InText) {
		App_.footer(InText.StdStr());
	}

	// Allow fallthrough so global options work with subcommands
	void SetFallthrough(bool InEnable = true) {
		App_.fallthrough(InEnable);
	}

protected:
	virtual void LoadYaml(const OaYaml::Node& InYaml) { (void)InYaml; }
	virtual void ApplyCliOverrides() {}

	template<typename T>
	CLI::Option* AddOption(const OaString& InFlag, T& InTarget, const OaString& InDesc) {
		auto* opt = App_.add_option(InFlag.StdStr(), InTarget, InDesc.StdStr());
		StoreOptionPtr(InFlag, opt);
		return opt;
	}

	CLI::Option* AddFlag(const OaString& InFlag, bool& InTarget, const OaString& InDesc) {
		auto* opt = App_.add_flag(InFlag.StdStr(), InTarget, InDesc.StdStr());
		StoreOptionPtr(InFlag, opt);
		return opt;
	}

	CLI::Option* AddMultiOption(const OaString& InFlag, OaVec<OaString>& InTarget, const OaString& InDesc) {
		auto* opt = App_.add_option(InFlag.StdStr(), InTarget, InDesc.StdStr());
		StoreOptionPtr(InFlag, opt);
		return opt;
	}

	template<typename T>
	CLI::Option* AddPositional(const OaString& InName, T& InTarget, const OaString& InDesc, bool InRequired = false) {
		auto* opt = App_.add_option(InName.StdStr(), InTarget, InDesc.StdStr());
		if (InRequired) {
			opt->required();
		}
		return opt;
	}

	[[nodiscard]] bool WasSet(const OaString& InFlag) const {
		auto it = Options_.Find(InFlag);
		if (it != Options_.End()) {
			return it->second->count() > 0;
		}
		return false;
	}

	void StoreOptionPtr(const OaString& InFlag, CLI::Option* InOpt) {
		auto it = Options_.Find(InFlag);
		if (it != Options_.End()) {
			it->second = InOpt;
		} else {
			Options_.Emplace(InFlag, InOpt);
		}
	}

private:
	OaHashMap<OaString, CLI::Option*> Options_;
};

#endif // OA_HAS_CLI11
