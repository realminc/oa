#include <gtest/gtest.h>

#include <Oa/Core/FnMatrix.h>
#include <Oa/Core/MatrixStorage.h>

static OaF32 OaTestMxAtF32(const OaMatrixStorage& InStorage, OaI64 InLinearIdx) {
	return reinterpret_cast<const OaF32*>(InStorage.HeapData())[InLinearIdx];
}

static void OaTestMxSetF32(OaMatrixStorage& InStorage, OaI64 InLinearIdx, OaF32 InValue) {
	reinterpret_cast<OaF32*>(InStorage.HeapData())[InLinearIdx] = InValue;
}

TEST(MatrixStorage, ZerosOnesShape) {
	OaMatrixShape shape = OaMatrixShape{2, 3};
	auto zero = OaMatrixStorage::Zeros(shape, OaScalarType::Float32);
	EXPECT_EQ(zero.NumElements(), 6);
	for (OaI64 idx = 0; idx < 6; ++idx) {
		EXPECT_FLOAT_EQ(OaTestMxAtF32(zero, idx), 0.0f);
	}
	auto one = OaMatrixStorage::Ones(shape, OaScalarType::Float32);
	for (OaI64 idx = 0; idx < 6; ++idx) {
		EXPECT_FLOAT_EQ(OaTestMxAtF32(one, idx), 1.0f);
	}
}

TEST(MatrixStorage, FromBytes) {
	OaMatrixShape shape = OaMatrixShape{3};
	OaF32 src[] = {1.0f, 2.0f, 4.0f};
	OaSpan<const OaU8> bytes(
		reinterpret_cast<const OaU8*>(src), sizeof(src));
	auto result = OaMatrixStorage::FromBytes(shape, OaScalarType::Float32, bytes);
	ASSERT_TRUE(result.IsOk());
	const OaMatrixStorage& storage = *result;
	EXPECT_FLOAT_EQ(OaTestMxAtF32(storage, 0), 1.0f);
	EXPECT_FLOAT_EQ(OaTestMxAtF32(storage, 2), 4.0f);
}

TEST(OaFnMatrix, AddMulScale) {
	OaMatrixShape shape = OaMatrixShape{2, 2};
	OaMatrixStorage storageA = OaMatrixStorage::Full(shape, OaScalarType::Float32, 1.0);
	OaMatrixStorage storageB = OaMatrixStorage::Full(shape, OaScalarType::Float32, 2.0);
	OaMatrix viewA = storageA.View();
	OaMatrix viewB = storageB.View();

	auto addResult = OaFnMatrix::Add(viewA, viewB);
	ASSERT_TRUE(addResult.IsOk());
	for (OaI64 idx = 0; idx < 4; ++idx) {
		EXPECT_FLOAT_EQ(OaTestMxAtF32(*addResult, idx), 3.0f);
	}

	auto mulResult = OaFnMatrix::Mul(viewA, viewB);
	ASSERT_TRUE(mulResult.IsOk());
	for (OaI64 idx = 0; idx < 4; ++idx) {
		EXPECT_FLOAT_EQ(OaTestMxAtF32(*mulResult, idx), 2.0f);
	}

	OaTestMxSetF32(storageA, 0, 5.0f);
	OaMatrix viewMut = storageA.View();
	EXPECT_TRUE(OaFnMatrix::ScaleInPlace(viewMut, 0.5).IsOk());
	EXPECT_FLOAT_EQ(OaTestMxAtF32(storageA, 0), 2.5f);
}

TEST(OaFnMatrix, SubDivNegInPlace) {
	OaMatrixShape shape = OaMatrixShape{4};
	OaMatrixStorage storageA = OaMatrixStorage::Zeros(shape, OaScalarType::Float32);
	OaMatrixStorage storageB = OaMatrixStorage::Zeros(shape, OaScalarType::Float32);
	for (OaI64 idx = 0; idx < 4; ++idx) {
		OaTestMxSetF32(storageA, idx, static_cast<OaF32>(idx + 1));
		OaTestMxSetF32(storageB, idx, 1.0f);
	}
	OaMatrix viewA = storageA.View();
	OaMatrix viewB = storageB.View();
	ASSERT_TRUE(OaFnMatrix::SubInPlace(viewA, viewB).IsOk());
	EXPECT_FLOAT_EQ(OaTestMxAtF32(storageA, 0), 0.0f);
	EXPECT_FLOAT_EQ(OaTestMxAtF32(storageA, 3), 3.0f);

	OaMatrixStorage storageC = OaMatrixStorage::Full(shape, OaScalarType::Float32, 8.0);
	OaMatrix viewC = storageC.View();
	ASSERT_TRUE(OaFnMatrix::DivInPlace(viewC, viewB).IsOk());
	for (OaI64 idx = 0; idx < 4; ++idx) {
		EXPECT_FLOAT_EQ(OaTestMxAtF32(storageC, idx), 8.0f);
	}

	ASSERT_TRUE(OaFnMatrix::NegInPlace(viewC).IsOk());
	for (OaI64 idx = 0; idx < 4; ++idx) {
		EXPECT_FLOAT_EQ(OaTestMxAtF32(storageC, idx), -8.0f);
	}
}

TEST(OaFnMatrix, MatMulTranspose) {
	OaMatrixStorage storageA = OaMatrixStorage::Zeros(OaMatrixShape{2, 3}, OaScalarType::Float32);
	OaMatrixStorage storageB = OaMatrixStorage::Zeros(OaMatrixShape{3, 2}, OaScalarType::Float32);
	// A = [[1,2,3],[4,5,6]], B = [[1,0],[0,1],[1,1]]  -> [[4,5],[10,11]]
	OaTestMxSetF32(storageA, 0, 1.0f);
	OaTestMxSetF32(storageA, 1, 2.0f);
	OaTestMxSetF32(storageA, 2, 3.0f);
	OaTestMxSetF32(storageA, 3, 4.0f);
	OaTestMxSetF32(storageA, 4, 5.0f);
	OaTestMxSetF32(storageA, 5, 6.0f);
	OaTestMxSetF32(storageB, 0, 1.0f);
	OaTestMxSetF32(storageB, 3, 1.0f);
	OaTestMxSetF32(storageB, 4, 1.0f);
	OaTestMxSetF32(storageB, 5, 1.0f);
	auto mulResult = OaFnMatrix::MatMulNt(storageA.View(), storageB.View());
	ASSERT_TRUE(mulResult.IsOk());
	EXPECT_EQ(mulResult->GetShape(), OaMatrixShape{2, 2});
	EXPECT_FLOAT_EQ(OaTestMxAtF32(*mulResult, 0), 4.0f);
	EXPECT_FLOAT_EQ(OaTestMxAtF32(*mulResult, 1), 5.0f);
	EXPECT_FLOAT_EQ(OaTestMxAtF32(*mulResult, 2), 10.0f);
	EXPECT_FLOAT_EQ(OaTestMxAtF32(*mulResult, 3), 11.0f);

	auto transResult = OaFnMatrix::Transpose(storageA.View());
	ASSERT_TRUE(transResult.IsOk());
	EXPECT_EQ(transResult->GetShape(), OaMatrixShape{3, 2});
	// A^T row-major: [[1,4],[2,5],[3,6]]
	EXPECT_FLOAT_EQ(OaTestMxAtF32(*transResult, 0), 1.0f);
	EXPECT_FLOAT_EQ(OaTestMxAtF32(*transResult, 1), 4.0f);
	EXPECT_FLOAT_EQ(OaTestMxAtF32(*transResult, 3), 5.0f);
	EXPECT_FLOAT_EQ(OaTestMxAtF32(*transResult, 5), 6.0f);
}

TEST(OaFnMatrix, ReductionsAndReshape) {
	OaMatrixShape shape2 = OaMatrixShape{2, 3};
	OaMatrixStorage storage = OaMatrixStorage::Zeros(shape2, OaScalarType::Float32);
	for (OaI64 idx = 0; idx < 6; ++idx) {
		OaTestMxSetF32(storage, idx, static_cast<OaF32>(idx + 1));
	}
	OaMatrix view = storage.View();

	auto sumResult = OaFnMatrix::Sum(view);
	ASSERT_TRUE(sumResult.IsOk());
	EXPECT_FLOAT_EQ(OaTestMxAtF32(*sumResult, 0), 21.0f);

	auto maxResult = OaFnMatrix::Max(view);
	ASSERT_TRUE(maxResult.IsOk());
	EXPECT_FLOAT_EQ(OaTestMxAtF32(*maxResult, 0), 6.0f);

	auto argMaxResult = OaFnMatrix::ArgMax(view);
	ASSERT_TRUE(argMaxResult.IsOk());
	EXPECT_EQ(argMaxResult->GetDtype(), OaScalarType::Int32);
	EXPECT_EQ(reinterpret_cast<const OaI32*>(argMaxResult->HeapData())[0], 5);

	auto colResult = OaFnMatrix::ReduceCols(view);
	ASSERT_TRUE(colResult.IsOk());
	EXPECT_EQ(colResult->GetShape(), OaMatrixShape{3});
	EXPECT_FLOAT_EQ(OaTestMxAtF32(*colResult, 0), 5.0f);
	EXPECT_FLOAT_EQ(OaTestMxAtF32(*colResult, 1), 7.0f);
	EXPECT_FLOAT_EQ(OaTestMxAtF32(*colResult, 2), 9.0f);

	auto meanResult = OaFnMatrix::ReduceMean(view);
	ASSERT_TRUE(meanResult.IsOk());
	EXPECT_FLOAT_EQ(OaTestMxAtF32(*meanResult, 0), 3.5f);

	auto reshapeResult = OaFnMatrix::ReshapeView(view, OaMatrixShape{6});
	ASSERT_TRUE(reshapeResult.IsOk());
	EXPECT_EQ(reshapeResult->GetShape(), OaMatrixShape{6});
	EXPECT_EQ(reshapeResult->HostBlock().Ptr, view.HostBlock().Ptr);
}

TEST(OaFnMatrix, Solve2x2) {
	// [2 0; 0 3] x = [4, 9]  -> x = [2, 3]
	OaMatrixStorage matA = OaMatrixStorage::Zeros(OaMatrixShape{2, 2}, OaScalarType::Float32);
	OaTestMxSetF32(matA, 0, 2.0f);
	OaTestMxSetF32(matA, 3, 3.0f);
	OaMatrixStorage vecB = OaMatrixStorage::Zeros(OaMatrixShape{2}, OaScalarType::Float32);
	OaTestMxSetF32(vecB, 0, 4.0f);
	OaTestMxSetF32(vecB, 1, 9.0f);
	auto solveResult = OaFnMatrix::Solve(matA.View(), vecB.View());
	ASSERT_TRUE(solveResult.IsOk());
	EXPECT_FLOAT_EQ(OaTestMxAtF32(*solveResult, 0), 2.0f);
	EXPECT_FLOAT_EQ(OaTestMxAtF32(*solveResult, 1), 3.0f);
}
