// OA CORE - Named OA filesystem locations
//
// Paths resolves application locations. oa::Path owns lexical path operations,
// while oa::Filesystem performs host filesystem I/O.

#pragma once

#include <oa/core/types.h>

namespace oa {

class Paths {
public:
	// Resolution: OA_ASSET_DIR > nearest source-tree sdk/asset > ./asset.
	[[nodiscard]] static oa::Path asset();
	[[nodiscard]] static oa::Path asset(oa::StringView inRelative);

	// Resolution: OA_VAR_DIR > nearest source-tree var > ./var.
	[[nodiscard]] static oa::Path var();
	[[nodiscard]] static oa::Path var(oa::StringView inRelative);

	// Optional datasets are never downloaded implicitly.
	// Resolution: OA_DATA_DIR > resolved var/data.
	[[nodiscard]] static oa::Path data();
	[[nodiscard]] static oa::Path data(oa::StringView inRelative);

	[[nodiscard]] static oa::Path current();
	[[nodiscard]] static oa::Path home();
	[[nodiscard]] static oa::Path temp();

private:
	[[nodiscard]] static oa::Path findSourceRoot();
};

} // namespace oa
