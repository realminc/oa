// oa::Cli<T> — native command-line configuration
//
// Precedence: struct defaults < YAML file < explicit command-line arguments.
// No CLI11/fmt types, exceptions, streams, or hosted string owners cross this
// public boundary.

#pragma once

#include <oa/core/yaml.h>
#include <oa/core/std/format.h>
#include <oa/core/std/limits.h>
#include <oa/core/std/typeTraits.h>
#include <oa/core/std/uniquePtr.h>

#include <stdio.h>

namespace oa {

namespace cliDetail {

template<typename T>
[[nodiscard]] bool parseValue(oa::StringView inText, T& outValue) noexcept {
	if constexpr (oa::IsSameV<T, oa::String>) {
		outValue = oa::String(inText);
		return true;
	} else if constexpr (oa::IsSameV<T, bool>) {
		if (inText == "true" or inText == "1" or inText == "on" or inText == "yes") {
			outValue = true;
			return true;
		}
		if (inText == "false" or inText == "0" or inText == "off" or inText == "no") {
			outValue = false;
			return true;
		}
		return false;
	} else if constexpr (oa::IsIntegralV<T>) {
		if constexpr (oa::Limits<T>::isSigned) {
			oa::I64 value = 0;
			if (not oa::parseI64(inText, value)
				or value < static_cast<oa::I64>(oa::Limits<T>::min())
				or value > static_cast<oa::I64>(oa::Limits<T>::max())) return false;
			outValue = static_cast<T>(value);
			return true;
		} else {
			oa::U64 value = 0;
			if (not oa::parseU64(inText, value)
				or value > static_cast<oa::U64>(oa::Limits<T>::max())) return false;
			outValue = static_cast<T>(value);
			return true;
		}
	} else if constexpr (oa::IsFloatingPointV<T>) {
		oa::F64 value = 0.0;
		if (not oa::parseF64(inText, value)) return false;
		outValue = static_cast<T>(value);
		return true;
	} else {
		static_assert(oa::IsSameV<T, void>, "unsupported oa::Cli option type");
		return false;
	}
}

} // namespace cliDetail

class CmdCli;

class OptCli {
public:
	OptCli* required() noexcept {
		required_ = true;
		return this;
	}

	[[nodiscard]] bool wasSet() const noexcept { return count_ != 0; }

private:
	friend class CmdCli;
	template<typename> friend class Cli;

	struct Alias {
		oa::String name;
		bool flagValue = true;
	};

	using ParseFn = bool (*)(void*, oa::StringView);

	[[nodiscard]] bool matches(oa::StringView inName, bool& outFlagValue) const noexcept {
		for (const Alias& alias : aliases_) {
			if (alias.name == inName) {
				outFlagValue = alias.flagValue;
				return true;
			}
		}
		return false;
	}

	oa::Vec<Alias> aliases_;
	oa::String description_;
	void* target_ = nullptr;
	ParseFn parse_ = nullptr;
	bool positional_ = false;
	bool flag_ = false;
	bool multi_ = false;
	bool required_ = false;
	oa::U32 count_ = 0;
};

class CmdCli {
public:
	CmdCli() = default;
	CmdCli(oa::String inName, oa::String inDescription)
		: name_(oa::move(inName)), description_(oa::move(inDescription)) {}

	template<typename T>
	OptCli* addOption(oa::StringView inNames, T& outTarget, oa::StringView inDescription) {
		auto option = oa::makeUnique<OptCli>();
		option->target_ = &outTarget;
		option->description_ = oa::String(inDescription);
		option->parse_ = [](void* inTarget, oa::StringView inValue) {
			return oa::cliDetail::parseValue(inValue, *static_cast<T*>(inTarget));
		};
		addAliases(*option, inNames);
		option->positional_ = not option->aliases_.empty()
			and (option->aliases_.front().name.empty()
				or option->aliases_.front().name[0] != '-');
		OptCli* result = option.get();
		options_.pushBack(oa::move(option));
		return result;
	}

	OptCli* addFlag(oa::StringView inNames, bool& outTarget, oa::StringView inDescription) {
		auto option = oa::makeUnique<OptCli>();
		option->target_ = &outTarget;
		option->description_ = oa::String(inDescription);
		option->flag_ = true;
		addAliases(*option, inNames);
		OptCli* result = option.get();
		options_.pushBack(oa::move(option));
		return result;
	}

	OptCli* addMultiOption(
		oa::StringView inNames,
		oa::Vec<oa::String>& outTarget,
		oa::StringView inDescription
	) {
		auto option = oa::makeUnique<OptCli>();
		option->target_ = &outTarget;
		option->description_ = oa::String(inDescription);
		option->multi_ = true;
		option->parse_ = [](void* inTarget, oa::StringView inValue) {
			static_cast<oa::Vec<oa::String>*>(inTarget)->pushBack(oa::String(inValue));
			return true;
		};
		addAliases(*option, inNames);
		OptCli* result = option.get();
		options_.pushBack(oa::move(option));
		return result;
	}

	template<typename T>
	OptCli* addPositional(
		oa::StringView inName,
		T& outTarget,
		oa::StringView inDescription,
		bool inRequired = false
	) {
		OptCli* option = addOption(inName, outTarget, inDescription);
		option->positional_ = true;
		if (inRequired) option->required();
		return option;
	}

	[[nodiscard]] const oa::String& getName() const noexcept { return name_; }
	[[nodiscard]] bool parsed() const noexcept { return parsed_; }

private:
	template<typename> friend class Cli;

	static void addAliases(OptCli& outOption, oa::StringView inNames) {
		oa::Usize begin = 0;
		while (begin <= inNames.size()) {
			const oa::Usize comma = inNames.find(',', begin);
			const oa::Usize end = comma == oa::StringView::Npos ? inNames.size() : comma;
			oa::StringView name = inNames.subStr(begin, end - begin);
			bool flagValue = true;
			const oa::StringView falseSuffix("{false}", 7U);
			if (name.size() >= falseSuffix.size()
				and name.subStr(name.size() - falseSuffix.size()) == falseSuffix) {
				name.removeSuffix(falseSuffix.size());
				flagValue = false;
			}
			outOption.aliases_.pushBack({oa::String(name), flagValue});
			if (comma == oa::StringView::Npos) break;
			begin = comma + 1U;
		}
	}

	[[nodiscard]] OptCli* findOption(oa::StringView inName, bool& outFlagValue) noexcept {
		for (oa::UniquePtr<OptCli>& option : options_) {
			if (not option->positional_ and option->matches(inName, outFlagValue)) {
				return option.get();
			}
		}
		return nullptr;
	}

	[[nodiscard]] OptCli* nextPositional() noexcept {
		for (oa::UniquePtr<OptCli>& option : options_) {
			if (option->positional_ and (option->multi_ or option->count_ == 0)) {
				return option.get();
			}
		}
		return nullptr;
	}

	[[nodiscard]] bool validateRequired(oa::String& outError) const {
		for (const oa::UniquePtr<OptCli>& option : options_) {
			if (option->required_ and option->count_ == 0) {
				outError = "missing required option: ";
				outError += option->aliases_.front().name;
				return false;
			}
		}
		return true;
	}

	oa::String name_;
	oa::String description_;
	oa::Vec<oa::UniquePtr<OptCli>> options_;
	bool parsed_ = false;
};

template<typename TConfig>
class Cli {
public:
	Cli(oa::StringView inName, oa::StringView inDescription)
		: root_(oa::String(inName), oa::String(inDescription)) {
		root_.addOption("-c,--config", configPath_, "YAML config file");
		root_.addOption("-v,--verbose", verbose_, "verbose level (0-3)");
	}

	virtual ~Cli() = default;

	bool parse(int inArgc, char** inArgv) {
		scanConfigPath(inArgc, inArgv);
		if (not configPath_.empty()) {
			try {
				loadYaml(oa::Yaml::loadFile(configPath_));
			} catch (const oa::Yaml::Exception& error) {
				::fprintf(stderr, "[OA CONFIG] YAML load failed: %s (using defaults)\n", error.what());
			}
		}
		if (not parseArguments(inArgc, inArgv)) return false;
		applyCliOverrides();
		return true;
	}

	[[nodiscard]] const TConfig& getConfig() const noexcept { return cfg_; }
	[[nodiscard]] TConfig& getConfig() noexcept { return cfg_; }
	[[nodiscard]] const oa::String& getConfigPath() const noexcept { return configPath_; }
	[[nodiscard]] oa::I32 getVerbose() const noexcept { return verbose_; }
	[[nodiscard]] bool helpRequested() const noexcept { return helpRequested_; }

	CmdCli* addSubcommand(oa::StringView inName, oa::StringView inDescription) {
		auto command = oa::makeUnique<CmdCli>(
			oa::String(inName), oa::String(inDescription));
		CmdCli* result = command.get();
		subcommands_.pushBack(oa::move(command));
		return result;
	}

	[[nodiscard]] bool gotSubcommand(oa::StringView inName) const noexcept {
		for (const oa::UniquePtr<CmdCli>& command : subcommands_) {
			if (command->name_ == inName) return command->parsed_;
		}
		return false;
	}

	[[nodiscard]] oa::String getSubcommand() const {
		for (const oa::UniquePtr<CmdCli>& command : subcommands_) {
			if (command->parsed_) return command->name_;
		}
		return {};
	}

	void requireSubcommand(oa::I32 inMin = 0, oa::I32 inMax = 0) noexcept {
		requiredSubcommandMin_ = inMin;
		requiredSubcommandMax_ = inMax;
	}

	void setEpilog(oa::StringView inText) { epilog_ = oa::String(inText); }
	void setFallthrough(bool inEnable = true) noexcept { fallthrough_ = inEnable; }

protected:
	virtual void loadYaml(const oa::Yaml::Node& inYaml) { (void)inYaml; }
	virtual void applyCliOverrides() {}

	template<typename T>
	OptCli* addOption(oa::StringView inNames, T& outTarget, oa::StringView inDescription) {
		return root_.addOption(inNames, outTarget, inDescription);
	}

	OptCli* addFlag(oa::StringView inNames, bool& outTarget, oa::StringView inDescription) {
		return root_.addFlag(inNames, outTarget, inDescription);
	}

	OptCli* addMultiOption(
		oa::StringView inNames,
		oa::Vec<oa::String>& outTarget,
		oa::StringView inDescription
	) {
		return root_.addMultiOption(inNames, outTarget, inDescription);
	}

	template<typename T>
	OptCli* addPositional(
		oa::StringView inName,
		T& outTarget,
		oa::StringView inDescription,
		bool inRequired = false
	) {
		return root_.addPositional(inName, outTarget, inDescription, inRequired);
	}

	[[nodiscard]] bool wasSet(oa::StringView inName) const noexcept {
		bool ignored = true;
		for (const oa::UniquePtr<OptCli>& option : root_.options_) {
			if (option->matches(inName, ignored)) return option->wasSet();
		}
		return false;
	}

	void scanConfigPath(int inArgc, char** inArgv) {
		for (int index = 1; index < inArgc; ++index) {
			oa::StringView argument(inArgv[index]);
			if ((argument == "--config" or argument == "-c") and index + 1 < inArgc) {
				configPath_ = inArgv[index + 1];
				return;
			}
			if (argument.size() > 9U and argument.subStr(0, 9) == "--config=") {
				configPath_ = oa::String(argument.subStr(9));
				return;
			}
			if (argument.size() > 3U and argument.subStr(0, 3) == "-c=") {
				configPath_ = oa::String(argument.subStr(3));
				return;
			}
		}
	}

	[[nodiscard]] bool parseArguments(int inArgc, char** inArgv) {
		helpRequested_ = false;
		CmdCli* selected = nullptr;
		for (int index = 1; index < inArgc; ++index) {
			oa::StringView argument(inArgv[index]);
			if (argument == "--help" or argument == "-h" or argument == "--help-all") {
				helpRequested_ = true;
				printHelp(selected, argument == "--help-all");
				return false;
			}
			if (selected == nullptr and not startsWithDash(argument)) {
				for (oa::UniquePtr<CmdCli>& command : subcommands_) {
					if (command->name_ == argument) {
						selected = command.get();
						command->parsed_ = true;
						break;
					}
				}
				if (selected != nullptr) continue;
			}
			if (startsWithDash(argument)) {
				oa::StringView name = argument;
				oa::StringView inlineValue;
				const oa::Usize equal = argument.find('=');
				const bool inlineProvided = equal != oa::StringView::Npos;
				if (inlineProvided) {
					name = argument.subStr(0, equal);
					inlineValue = argument.subStr(equal + 1U);
				}
				bool flagValue = true;
				OptCli* option = selected ? selected->findOption(name, flagValue) : nullptr;
				if (option == nullptr and (selected == nullptr or fallthrough_)) {
					option = root_.findOption(name, flagValue);
				}
				if (option == nullptr) return fail("unknown option", name);
				if (option->flag_) {
					if (inlineProvided) return fail("flag does not take a value", name);
					*static_cast<bool*>(option->target_) = flagValue;
					++option->count_;
					continue;
				}
				oa::StringView value = inlineValue;
				if (not inlineProvided) {
					if (index + 1 >= inArgc) return fail("option requires a value", name);
					value = inArgv[++index];
				}
				if (not option->parse_(option->target_, value)) return fail("invalid option value", name);
				++option->count_;
				continue;
			}
			CmdCli* command = selected != nullptr ? selected : &root_;
			OptCli* positional = command->nextPositional();
			if (positional == nullptr) return fail("unexpected argument", argument);
			if (not positional->parse_(positional->target_, argument)) {
				return fail("invalid positional value", argument);
			}
			++positional->count_;
		}
		const oa::I32 selectedCount = selected == nullptr ? 0 : 1;
		if (selectedCount < requiredSubcommandMin_
			or (requiredSubcommandMax_ > 0 and selectedCount > requiredSubcommandMax_)) {
			return fail("required subcommand missing", {});
		}
		oa::String requiredError;
		if (not root_.validateRequired(requiredError)) return fail(requiredError.view(), {});
		if (selected != nullptr and not selected->validateRequired(requiredError)) {
			return fail(requiredError.view(), {});
		}
		return true;
	}

	TConfig cfg_{};
	oa::String configPath_;
	oa::I32 verbose_ = 0;

private:
	[[nodiscard]] static bool startsWithDash(oa::StringView inText) noexcept {
		return not inText.empty() and inText[0] == '-';
	}

	[[nodiscard]] bool fail(oa::StringView inMessage, oa::StringView inDetail) const {
		::fprintf(stderr, "%.*s", static_cast<int>(inMessage.size()), inMessage.data());
		if (not inDetail.empty()) {
			::fprintf(stderr, ": %.*s", static_cast<int>(inDetail.size()), inDetail.data());
		}
		::fprintf(stderr, "\nUse --help for usage.\n");
		return false;
	}

	static void printOption(const OptCli& inOption) {
		::fprintf(stderr, "  ");
		for (oa::Usize index = 0; index < inOption.aliases_.size(); ++index) {
			if (index != 0) ::fprintf(stderr, ", ");
			const oa::String& name = inOption.aliases_[index].name;
			::fprintf(stderr, "%.*s", static_cast<int>(name.size()), name.data());
		}
		if (not inOption.flag_ and not inOption.positional_) ::fprintf(stderr, " <value>");
		::fprintf(stderr, "\n      %.*s%s\n",
			static_cast<int>(inOption.description_.size()), inOption.description_.data(),
			inOption.required_ ? " (required)" : "");
	}

	void printCommand(const CmdCli& inCommand) const {
		for (const oa::UniquePtr<OptCli>& option : inCommand.options_) printOption(*option);
	}

	void printHelp(const CmdCli* inSelected, bool inAll) const {
		const CmdCli& command = inSelected == nullptr ? root_ : *inSelected;
		::fprintf(stderr, "%.*s — %.*s\n",
			static_cast<int>(command.name_.size()), command.name_.data(),
			static_cast<int>(command.description_.size()), command.description_.data());
		printCommand(command);
		if (inSelected == nullptr and not subcommands_.empty()) {
			::fprintf(stderr, "\nCommands:\n");
			for (const oa::UniquePtr<CmdCli>& subcommand : subcommands_) {
				::fprintf(stderr, "  %.*s\n      %.*s\n",
					static_cast<int>(subcommand->name_.size()), subcommand->name_.data(),
					static_cast<int>(subcommand->description_.size()), subcommand->description_.data());
				if (inAll) printCommand(*subcommand);
			}
		}
		if (not epilog_.empty()) {
			::fprintf(stderr, "\n%.*s\n", static_cast<int>(epilog_.size()), epilog_.data());
		}
	}

	CmdCli root_;
	oa::Vec<oa::UniquePtr<CmdCli>> subcommands_;
	oa::String epilog_;
	oa::I32 requiredSubcommandMin_ = 0;
	oa::I32 requiredSubcommandMax_ = 0;
	bool fallthrough_ = true;
	bool helpRequested_ = false;
};

} // namespace oa
