#include "oaStdTest.h"

#include <filesystem>

TEST(Filesystem, ExistsCurrentDir) {
	EXPECT_TRUE(oa::Filesystem::exists(oa::Path(".")));
}

TEST(Filesystem, TempDirRoundTrip) {
	const oa::Path sub = oa::Paths::temp() / "oa_oastd_fs_probe";
	(void)oa::Filesystem::removeDirectory(sub, true);
	ASSERT_TRUE(oa::Filesystem::createDirectories(sub).isOk());
	EXPECT_TRUE(oa::Filesystem::isDirectory(sub));
	EXPECT_TRUE(oa::Filesystem::exists(sub));
	EXPECT_TRUE(oa::Filesystem::removeDirectory(sub, true).isOk());
	EXPECT_FALSE(oa::Filesystem::exists(sub));
}

TEST(Filesystem, AbsoluteCurrentDirectory) {
	auto absolute = oa::Filesystem::absolute(oa::Path("."));
	ASSERT_TRUE(absolute.isOk());
	EXPECT_TRUE(absolute->isAbsolute());
	EXPECT_TRUE(oa::Filesystem::isDirectory(*absolute));
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

TEST(Path, RootDotAndDotDotSemantics) {
	EXPECT_EQ(oa::Path("/").parentPath(), oa::Path("/"));
	EXPECT_TRUE(oa::Path("file").parentPath().empty());
	EXPECT_EQ(oa::Path("/a").parentPath(), oa::Path("/"));
	EXPECT_EQ(oa::Path("a/").parentPath(), oa::Path("a"));
	EXPECT_EQ(oa::Path("a/b/").parentPath(), oa::Path("a/b"));
	EXPECT_EQ(oa::Path("a/./b/../c").lexicallyNormal(), oa::Path("a/c"));
	EXPECT_EQ(oa::Path("../../a").lexicallyNormal(), oa::Path("../../a"));
	EXPECT_EQ(oa::Path("/../../a").lexicallyNormal(), oa::Path("/a"));
	EXPECT_EQ(oa::Path("archive.tar.gz").stem(), oa::Path("archive.tar"));
	EXPECT_EQ(oa::Path("archive.tar.gz").extension(), oa::Path(".gz"));
	EXPECT_TRUE(oa::Path(".profile").extension().empty());
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

TEST(PathVsStd, StringMatchesStdFilesystemPath) {
	const std::filesystem::path st("foo/bar/baz.txt");
	oa::Path oa(st.string().c_str());
	stdEchoCurrentTest();
	stdExpectGotSize("path string length (match)", st.string().size(), oa.string().size());
	EXPECT_EQ(testStdString(oa.string()), st.string());
	EXPECT_EQ(testStdString(oa.genericString()), st.generic_string());
	EXPECT_EQ(testStdString(oa.filename().string()), st.filename().string());
	EXPECT_EQ(testStdString(oa.parentPath().string()), st.parent_path().string());
}

TEST(PathVsStd, TimedAppendWallUs) {
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
