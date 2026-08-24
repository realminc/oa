#include "oaStdTest.h"

#include <filesystem>

TEST(StdFilesystem, ExistsCurrentDir) {
	EXPECT_TRUE(oa::StdFilesystem::exists(oa::Path(".")));
}

TEST(StdFilesystem, TempDirRoundTrip) {
	std::error_code bec;
	const std::filesystem::path base = std::filesystem::temp_directory_path(bec);
	ASSERT_FALSE(bec);
	const oa::Path sub(std::filesystem::path(base) / "oa_oastd_fs_probe");
	(void)oa::StdFilesystem::removeAll(sub);
	ASSERT_TRUE(oa::StdFilesystem::createDirectories(sub));
	EXPECT_TRUE(oa::StdFilesystem::isDirectory(sub));
	EXPECT_TRUE(oa::StdFilesystem::exists(sub));
	EXPECT_TRUE(oa::StdFilesystem::removeAll(sub));
	EXPECT_FALSE(oa::StdFilesystem::exists(sub));
}

TEST(StdFilesystem, EquivalentDotAndCurrentPath) {
	std::error_code ec;
	const std::filesystem::path cur = std::filesystem::current_path(ec);
	ASSERT_FALSE(ec);
	EXPECT_TRUE(oa::StdFilesystem::equivalent(oa::Path("."), oa::Path(cur)));
}

TEST(StdFilesystem, IsSymlinkFalseForDot) {
	EXPECT_FALSE(oa::StdFilesystem::isSymlink(oa::Path(".")));
}

TEST(Path, LexicallyNormalCollapsesComponents) {
	oa::Path messy("a/b/../c");
	EXPECT_EQ(messy.lexicallyNormal().string(), oa::Path("a/c").string());
}

TEST(Path, AppendFilename) {
	oa::Path root("a");
	root /= oa::Path("b.txt");
	EXPECT_EQ(root.filename().string(), "b.txt");
}

TEST(Path, EqualAndSwap) {
	oa::Path a("x");
	oa::Path b("x");
	oa::Path c("y");
	EXPECT_TRUE(a == b);
	EXPECT_FALSE(a == c);
	a.swap(c);
	EXPECT_EQ(a.string(), "y");
	EXPECT_EQ(c.string(), "x");
}

TEST(StdPathVsStd, StringMatchesStdFilesystemPath) {
	const std::filesystem::path st("foo/bar/baz.txt");
	oa::Path oa(st);
	stdEchoCurrentTest();
	stdExpectGotSize("path string length (match)", st.string().size(), oa.string().size());
	EXPECT_EQ(oa.string(), st.string());
	EXPECT_EQ(oa.genericString(), st.generic_string());
	EXPECT_EQ(oa.filename().string(), st.filename().string());
	EXPECT_EQ(oa.parentPath().string(), st.parent_path().string());
}

TEST(StdPathVsStd, TimedAppendWallUs) {
	constexpr int kIters = 50'000;
	const auto t0 = oa::highResolutionNow();
	for (int i = 0; i < kIters; ++i) {
		oa::Path p("base");
		p /= oa::Path("segment");
		p /= oa::Path("file.txt");
		(void)p.string();
	}
	const auto t1 = oa::highResolutionNow();
	for (int i = 0; i < kIters; ++i) {
		std::filesystem::path p("base");
		p /= "segment";
		p /= "file.txt";
		(void)p.string();
	}
	const auto t2 = oa::highResolutionNow();
	stdReportCompareSequentialRuns(
		"oa::Path append+String x50k", t0, t1, "std::filesystem::path append+string x50k", t2);
}
