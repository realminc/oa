#include "lunarLander3dViewerSource.h"

#include <oa/core/log.h>
#include <oa/ui/viewer.h>

namespace {

[[nodiscard]] oa::I32 parseSampleCount(
	int inArgc,
	char** inArgv,
	oa::U32& outSampleCount) {
	outSampleCount = 1U;
	for (int index = 1; index < inArgc; ++index) {
		const oa::StringView argument(inArgv[index]);
		if (argument == "--help" or argument == "-h") {
			OA_CLI("usage: TutorialRlLunarLander3dViewer [--msaa 1|2|4|8|16|32|64]");
			return 1;
		}
		if (argument != "--msaa" or index + 1 >= inArgc) {
			OA_CLI("Unknown or incomplete argument: {}", inArgv[index]);
			return 2;
		}
		const oa::StringView value(inArgv[++index]);
		oa::U32 parsed = 0U;
		for (char digit : value) {
			if (digit < '0' or digit > '9'
				or parsed > (64U - static_cast<oa::U32>(digit - '0')) / 10U) {
				OA_CLI("Invalid --msaa value: {}", inArgv[index]);
				return 2;
			}
			parsed = parsed * 10U + static_cast<oa::U32>(digit - '0');
		}
		if (value.empty()
			or (parsed != 1U and parsed != 2U and parsed != 4U
				and parsed != 8U and parsed != 16U
				and parsed != 32U and parsed != 64U)) {
			OA_CLI("Invalid --msaa value: {}", inArgv[index]);
			return 2;
		}
		outSampleCount = parsed;
	}
	return 0;
}

} // namespace

int main(int argc, char** argv) {
	oa::U32 sampleCount = 1U;
	const oa::I32 parseResult = parseSampleCount(argc, argv, sampleCount);
	if (parseResult != 0) return parseResult == 1 ? 0 : 1;
	LunarLander3dViewerSource source(sampleCount);
	oa::Viewer viewer({
		.mode = oa::ViewerMode::Live,
		.liveSource = &source,
		.title = "OA · Lunar Lander 3D",
		.width = 1280U,
		.height = 720U,
		.showHelp = true,
		.showStats = false,
		.showTimeline = false,
		.vsync = true,
		.presentFilter = oa::Filter::Linear,
	});
	return viewer.run().isOk() ? 0 : 1;
}
