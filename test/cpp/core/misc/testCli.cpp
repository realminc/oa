#include <gtest/gtest.h>

#include <oa/core/cli.h>

#include <initializer_list>

namespace {

template<typename Parser>
bool parseArgs(Parser& inParser, std::initializer_list<const char*> inArguments) {
	oa::Vector<oa::String> owned;
	owned.reserve(inArguments.size());
	for (const char* argument : inArguments) owned.pushBack(oa::String(argument));
	oa::Vector<char*> pointers;
	pointers.reserve(owned.size());
	for (oa::String& argument : owned) {
		pointers.pushBack(const_cast<char*>(argument.cStr()));
	}
	return inParser.parse(static_cast<int>(pointers.size()), pointers.data());
}

struct CliConfig {
	oa::I32 count = 1;
	oa::F32 rate = 0.0F;
	oa::String path;
	bool feature = false;
	oa::Vector<oa::String> labels;
};

class TestCli final : public oa::Cli<CliConfig> {
public:
	TestCli() : oa::Cli<CliConfig>("testCli", "native parser test") {
		addOption("--count,-n", cfg_.count, "item count");
		addOption("--rate", cfg_.rate, "rate");
		addOption("--path", cfg_.path, "path");
		addFlag("--feature,--no-feature{false}", cfg_.feature, "feature toggle");
		addMultiOption("--label", cfg_.labels, "repeatable label");
	}
};

struct CommandConfig {
	oa::String input;
	oa::String output;
};

class CommandCli final : public oa::Cli<CommandConfig> {
public:
	CommandCli() : oa::Cli<CommandConfig>("commandCli", "subcommand parser test") {
		oa::CmdCli* pack = addSubcommand("pack", "pack one input");
		pack->addOption("input", cfg_.input, "input path")->required();
		pack->addOption("--output,-o", cfg_.output, "output path")->required();
		requireSubcommand(1, 1);
	}
};

class StableHandleCli final : public oa::Cli<CommandConfig> {
public:
	StableHandleCli() : oa::Cli<CommandConfig>("stableCli", "stable handle test") {
		oa::CmdCli* first = addSubcommand("first", "first command");
		addSubcommand("second", "second command");
		first->addOption("input", cfg_.input, "input path")->required();

		oa::OptCli* output = addOption("--output", cfg_.output, "output path");
		for (oa::I32 index = 0; index < 16; ++index) {
			addFlag("--unused", unused_[index], "unused growth option");
		}
		output->required();
	}

private:
	bool unused_[16]{};
};

TEST(Cli, ParsesAliasesTypedValuesFlagsAndRepeatedOptions) {
	TestCli parser;
	ASSERT_TRUE(parseArgs(parser, {
		"testCli", "-n", "42", "--rate=-0.125", "--feature",
		"--no-feature", "--path=", "--label", "one", "--label=two"}));
	const CliConfig& config = parser.getConfig();
	EXPECT_EQ(config.count, 42);
	EXPECT_FLOAT_EQ(config.rate, -0.125F);
	EXPECT_FALSE(config.feature);
	EXPECT_TRUE(config.path.empty());
	ASSERT_EQ(config.labels.size(), 2U);
	EXPECT_EQ(config.labels[0], "one");
	EXPECT_EQ(config.labels[1], "two");
}

TEST(Cli, KeepsCommandAndOptionHandlesStableAcrossGrowth) {
	StableHandleCli parser;
	ASSERT_TRUE(parseArgs(parser, {"stableCli", "first", "in.bin", "--output=out.bin"}));
	EXPECT_EQ(parser.getConfig().input, "in.bin");
	EXPECT_EQ(parser.getConfig().output, "out.bin");
}

TEST(Cli, ParsesSubcommandPositionalsAndRequiredOptions) {
	CommandCli parser;
	ASSERT_TRUE(parseArgs(parser, {
		"commandCli", "pack", "input.bin", "-o", "output.oad"}));
	EXPECT_TRUE(parser.gotSubcommand("pack"));
	EXPECT_EQ(parser.getSubcommand(), "pack");
	EXPECT_EQ(parser.getConfig().input, "input.bin");
	EXPECT_EQ(parser.getConfig().output, "output.oad");
}

TEST(Cli, RejectsMissingRequiredOptionAndUnknownOption) {
	CommandCli missing;
	EXPECT_FALSE(parseArgs(missing, {"commandCli", "pack", "input.bin"}));

	TestCli unknown;
	EXPECT_FALSE(parseArgs(unknown, {"testCli", "--unknown"}));
}

TEST(Cli, DistinguishesHelpFromParseFailure) {
	TestCli parser;
	EXPECT_FALSE(parseArgs(parser, {"testCli", "--help"}));
	EXPECT_TRUE(parser.helpRequested());
}

} // namespace
