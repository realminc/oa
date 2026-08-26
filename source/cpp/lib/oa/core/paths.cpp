#include <oa/core/paths.h>

#include <oa/core/filesystem.h>

#include <stdlib.h>

#ifdef _WIN32
	#include <direct.h>
#else
	#include <unistd.h>
#endif

namespace {

[[nodiscard]] const char* environment(const char* inName) noexcept {
	const char* value = ::getenv(inName);
	return value != nullptr and value[0] != '\0' ? value : nullptr;
}

} // namespace

oa::Path oa::Paths::asset() {
	if (const char* value = environment("OA_ASSET_DIR")) return oa::Path(value);
	const oa::Path sourceRoot = findSourceRoot();
	return sourceRoot.empty() ? oa::Path("asset") : sourceRoot / "sdk" / "asset";
}

oa::Path oa::Paths::asset(oa::StringView inRelative) {
	return asset() / oa::Path(inRelative);
}

oa::Path oa::Paths::var() {
	if (const char* value = environment("OA_VAR_DIR")) return oa::Path(value);
	const oa::Path sourceRoot = findSourceRoot();
	return sourceRoot.empty() ? oa::Path("var") : sourceRoot / "var";
}

oa::Path oa::Paths::var(oa::StringView inRelative) {
	return var() / oa::Path(inRelative);
}

oa::Path oa::Paths::data() {
	if (const char* value = environment("OA_DATA_DIR")) return oa::Path(value);
	return var("data");
}

oa::Path oa::Paths::data(oa::StringView inRelative) {
	return data() / oa::Path(inRelative);
}

oa::Path oa::Paths::current() {
#ifdef _WIN32
	char* directory = ::_getcwd(nullptr, 0);
#else
	char* directory = ::getcwd(nullptr, 0);
#endif
	if (directory == nullptr) return {};
	oa::Path result(directory);
	::free(directory);
	return result;
}

oa::Path oa::Paths::home() {
#ifdef _WIN32
	const char* value = environment("USERPROFILE");
#else
	const char* value = environment("HOME");
#endif
	return value != nullptr ? oa::Path(value) : oa::Path{};
}

oa::Path oa::Paths::temp() {
	if (const char* value = environment("TMPDIR")) return oa::Path(value);
	if (const char* value = environment("TEMP")) return oa::Path(value);
	if (const char* value = environment("TMP")) return oa::Path(value);
#ifdef _WIN32
	return current();
#else
	return oa::Path("/tmp");
#endif
}

oa::Path oa::Paths::findSourceRoot() {
	oa::Path currentPath = current();
	while (not currentPath.empty()) {
		if (oa::Filesystem::isDirectory(currentPath / "sdk" / "asset")) return currentPath;
		const oa::Path parent = currentPath.parentPath();
		if (parent.empty() or parent == currentPath) break;
		currentPath = parent;
	}
	return {};
}
