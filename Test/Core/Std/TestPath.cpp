#include "OaStdTest.h"

#include <filesystem>

TEST(OaStdFilesystem, ExistsCurrentDir) {
	EXPECT_TRUE(OaStdFilesystem::Exists(OaStdPath(".")));
}

TEST(OaStdFilesystem, TempDirRoundTrip) {
	std::error_code bec;
	const std::filesystem::path base = std::filesystem::temp_directory_path(bec);
	ASSERT_FALSE(bec);
	const OaStdPath sub(std::filesystem::path(base) / "oa_oastd_fs_probe");
	(void)OaStdFilesystem::RemoveAll(sub);
	ASSERT_TRUE(OaStdFilesystem::CreateDirectories(sub));
	EXPECT_TRUE(OaStdFilesystem::IsDirectory(sub));
	EXPECT_TRUE(OaStdFilesystem::Exists(sub));
	EXPECT_TRUE(OaStdFilesystem::RemoveAll(sub));
	EXPECT_FALSE(OaStdFilesystem::Exists(sub));
}

TEST(OaStdFilesystem, EquivalentDotAndCurrentPath) {
	std::error_code ec;
	const std::filesystem::path cur = std::filesystem::current_path(ec);
	ASSERT_FALSE(ec);
	EXPECT_TRUE(OaStdFilesystem::Equivalent(OaStdPath("."), OaStdPath(cur)));
}

TEST(OaStdFilesystem, IsSymlinkFalseForDot) {
	EXPECT_FALSE(OaStdFilesystem::IsSymlink(OaStdPath(".")));
}

TEST(OaStdPath, LexicallyNormalCollapsesComponents) {
	OaStdPath messy("a/b/../c");
	EXPECT_EQ(messy.LexicallyNormal().String(), OaStdPath("a/c").String());
}

TEST(OaStdPath, AppendFilename) {
	OaStdPath root("a");
	root /= OaStdPath("b.txt");
	EXPECT_EQ(root.Filename().String(), "b.txt");
}

TEST(OaStdPath, EqualAndSwap) {
	OaStdPath a("x");
	OaStdPath b("x");
	OaStdPath c("y");
	EXPECT_TRUE(a == b);
	EXPECT_FALSE(a == c);
	a.Swap(c);
	EXPECT_EQ(a.String(), "y");
	EXPECT_EQ(c.String(), "x");
}

TEST(OaStdPathVsStd, StringMatchesStdFilesystemPath) {
	const std::filesystem::path st("foo/bar/baz.txt");
	OaStdPath oa(st);
	OaStdEchoCurrentTest();
	OaStdExpectGotSize("path string length (match)", st.string().size(), oa.String().Size());
	EXPECT_EQ(oa.String(), st.string());
	EXPECT_EQ(oa.GenericString(), st.generic_string());
	EXPECT_EQ(oa.Filename().String(), st.filename().string());
	EXPECT_EQ(oa.ParentPath().String(), st.parent_path().string());
}

TEST(OaStdPathVsStd, TimedAppendWallUs) {
	constexpr int kIters = 50'000;
	const auto t0 = OaHighResolutionNow();
	for (int i = 0; i < kIters; ++i) {
		OaStdPath p("base");
		p /= OaStdPath("segment");
		p /= OaStdPath("file.txt");
		(void)p.String();
	}
	const auto t1 = OaHighResolutionNow();
	for (int i = 0; i < kIters; ++i) {
		std::filesystem::path p("base");
		p /= "segment";
		p /= "file.txt";
		(void)p.string();
	}
	const auto t2 = OaHighResolutionNow();
	OaStdReportCompareSequentialRuns(
		"OaStdPath append+String x50k", t0, t1, "std::filesystem::path append+string x50k", t2);
}
