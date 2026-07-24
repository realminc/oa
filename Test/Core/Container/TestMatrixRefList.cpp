#include <gtest/gtest.h>

#include <Oa/Core/Filesystem.h>
#include <Oa/Core/Paths.h>
#include <Oa/Core/MatrixList.h>
#include <Oa/Core/MatrixRef.h>
#include <Oa/Core/MatrixStorage.h>

#include <atomic>
#include <chrono>
#include <cstring>

static std::atomic<OaU64> gOaMatrixListTestDirSeq{0};

static OaPath OaMatrixListTestMakeWorkDir() {
	OaPath tmp = OaPaths::Temp();
	const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
	OaString name = OaString("oa_matrix_ref_list_") + std::to_string(static_cast<long long>(++gOaMatrixListTestDirSeq)) + "_"
		+ std::to_string(static_cast<long long>(tick));
	return tmp / OaPath(name);
}

class MatrixRefListFs : public ::testing::Test {
protected:
	OaPath WorkDir_;

	void SetUp() override {
		WorkDir_ = OaMatrixListTestMakeWorkDir();
		ASSERT_TRUE(OaFilesystem::CreateDirectories(WorkDir_).IsOk());
	}

	void TearDown() override {
		(void)OaFilesystem::RemoveDirectory(WorkDir_, true);
	}
};

TEST(OaMatrixRef, EvalCopiesHostRowMajor) {
	OaMatrixStorage stor = OaMatrixStorage::Full(OaMatrixShape{2, 2}, OaScalarType::Float32, 3.5f);
	const OaMatrix view = stor.View();
	OaMatrixRef ref(view);
	auto out = ref.Eval();
	ASSERT_TRUE(out.IsOk());
	EXPECT_EQ(out->GetShape(), view.GetShape());
	EXPECT_EQ(out->GetDtype(), OaScalarType::Float32);
	const auto* heap = reinterpret_cast<const OaF32*>(out->HeapData());
	for (OaI64 idx = 0; idx < 4; ++idx) {
		EXPECT_FLOAT_EQ(heap[idx], 3.5f);
	}
}

TEST(OaMatrixRef, WritableMutatesVersusReadOnly) {
	OaMatrixStorage stor = OaMatrixStorage::Zeros(OaMatrixShape{2}, OaScalarType::Float32);
	OaMatrix mutView = stor.View();
	OaMatrixRef mutRef(mutView);
	ASSERT_NE(mutRef.DataPtr(), nullptr);
	auto* floats = static_cast<OaF32*>(mutRef.DataPtr());
	floats[0] = 9.0f;
	EXPECT_FLOAT_EQ(reinterpret_cast<OaF32*>(stor.HeapData())[0], 9.0f);

	OaMatrixStorage storConst = OaMatrixStorage::Zeros(OaMatrixShape{1}, OaScalarType::Float32);
	const OaMatrix readView = storConst.View();
	OaMatrixRef readOnlyRef(readView);
	EXPECT_EQ(readOnlyRef.DataPtr(), nullptr);
}

TEST(OaMatrixList, AddGetHasRemoveForEachOrder) {
	OaMatrixList list;
	EXPECT_EQ(list.Size(), 0u);
	EXPECT_FALSE(list.Has("a"));

	list.Add("second", OaMatrixStorage::Full(OaMatrixShape{2}, OaScalarType::Float32, 2.0f));
	list.Add("first", OaMatrixStorage::Full(OaMatrixShape{1}, OaScalarType::Float32, 1.0f));
	EXPECT_EQ(list.Size(), 2u);
	EXPECT_TRUE(list.Has("first"));

	auto missing = list.Get("nope");
	EXPECT_FALSE(missing.IsOk());

	auto gotFirst = list.Get("first");
	ASSERT_TRUE(gotFirst.IsOk());
	EXPECT_EQ(gotFirst->GetShape(), OaMatrixShape{1});
	EXPECT_FLOAT_EQ(reinterpret_cast<const OaF32*>(gotFirst->HostBlock().Ptr)[0], 1.0f);

	OaVec<OaString> order;
	list.ForEach(OaFunc<void(OaStringView, OaMatrix)>([&order](OaStringView InName, OaMatrix InMat) {
		(void)InMat;
		order.PushBack(OaString(InName));
	}));
	ASSERT_EQ(order.Size(), 2u);
	EXPECT_EQ(order[0], OaString("second"));
	EXPECT_EQ(order[1], OaString("first"));

	list.Remove("second");
	EXPECT_FALSE(list.Has("second"));
	EXPECT_EQ(list.Size(), 1u);
}

TEST(OaMatrixList, AddReplaceKeepsOrder) {
	OaMatrixList list;
	list.Add("key", OaMatrixStorage::Full(OaMatrixShape{1}, OaScalarType::Float32, 1.0f));
	list.Add("other", OaMatrixStorage::Full(OaMatrixShape{1}, OaScalarType::Float32, 2.0f));
	list.Add("key", OaMatrixStorage::Full(OaMatrixShape{1}, OaScalarType::Float32, 99.0f));

	OaVec<OaString> order;
	list.ForEach(OaFunc<void(OaStringView, OaMatrix)>([&order](OaStringView InName, OaMatrix InMat) {
		(void)InMat;
		order.PushBack(OaString(InName));
	}));
	ASSERT_EQ(order.Size(), 2u);
	EXPECT_EQ(order[0], OaString("key"));
	EXPECT_EQ(order[1], OaString("other"));

	auto got = list.Get("key");
	ASSERT_TRUE(got.IsOk());
	EXPECT_FLOAT_EQ(reinterpret_cast<const OaF32*>(got->HostBlock().Ptr)[0], 99.0f);
}

TEST_F(MatrixRefListFs, SaveLoadRoundTrip) {
	OaMatrixList list;
	list.Add("m", OaMatrixStorage::Full(OaMatrixShape{2, 3}, OaScalarType::Float32, 7.0f));

	OaPath path = WorkDir_ / "matrices.oaml";
	ASSERT_TRUE(list.SaveToFile(OaStringView(path.CStr())).IsOk());

	OaMatrixList loaded;
	ASSERT_TRUE(loaded.LoadFromFile(OaStringView(path.CStr())).IsOk());
	EXPECT_EQ(loaded.Size(), 1u);

	auto got = loaded.Get("m");
	ASSERT_TRUE(got.IsOk());
	EXPECT_EQ(got->GetShape(), OaMatrixShape{2, 3});
	const auto* heap = reinterpret_cast<const OaF32*>(got->HostBlock().Ptr);
	for (OaI64 idx = 0; idx < 6; ++idx) {
		EXPECT_FLOAT_EQ(heap[idx], 7.0f);
	}
}
