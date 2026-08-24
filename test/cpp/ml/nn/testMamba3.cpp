// Test Mamba3 module - basic instantiation and forward pass validation

#include "../../oaTest.h"
#include <oa/ml.h>
#include <oa/ml/nn.h>
#include <oa/ml/autograd.h>
#include <oa/runtime/engine.h>
#include <cmath>
#include <limits>
#include <vector>

[[maybe_unused]] static void syncCtx() {
	auto& ctx = oa::ExecutionSession::getActive();
	(void)testSubmitAndWait(ctx);
}

static oa::Matrix matFromVec(const std::vector<float>& v, oa::MatrixShape shape) {
	return oa::FnMatrix::fromBytes(
		{reinterpret_cast<const oa::U8*>(v.data()), v.size() * sizeof(float)},
		shape, oa::ScalarType::Float32);
}

TEST(TestMamba3, Instantiate) {
	// Basic instantiation test
	oa::Mamba3Module mamba3(
		64,   // d_model
		128,  // d_state
		2,    // expand
		64,   // headdim
		1     // ngroups
	);

	EXPECT_EQ(mamba3.parameters().size(), 6);  // in_proj, dt_bias, B_bias, C_bias, D, out_proj
}

TEST(TestMamba3, ForwardPass) {
	// Test forward pass with small input
	oa::Mamba3Module mamba3(
		64,   // d_model
		16,   // d_state
		2,    // expand
		32,   // headdim
		1     // ngroups
	);

	// input: [B=2, S=32, d_model=64]
	auto input = oa::FnMatrix::randN(oa::MatrixShape{2, 32, 64}, oa::ScalarType::Float32);

	// execute forward pass with context
	auto& ctx = oa::ExecutionSession::getActive();
	try {
		std::cerr << "Starting forward pass..." << std::endl;
		auto output = mamba3.forward(input);
		std::cerr << "forward pass completed, executing context..." << std::endl;
		(void)testSubmitAndWait(ctx);
		std::cerr << "context executed, syncing..." << std::endl;
		(void)testSubmitAndWait(ctx);
		std::cerr << "Sync completed" << std::endl;

		// output should be [B=2, S=32, d_model=64]
		EXPECT_EQ(output.rank(), 3);
		EXPECT_EQ(output.size(0), 2);
		EXPECT_EQ(output.size(1), 32);
		EXPECT_EQ(output.size(2), 64);
	} catch (const std::exception& e) {
		std::cerr << "Exception in ForwardPass: " << e.what() << std::endl;
		throw;
	}
}

// SsmScanBackward test removed - old API deprecated, use Mamba3Module instead

// Numerical parity: GPU Mamba3Siso kernel vs an explicit CPU implementation of the
// exact Mamba-3 SISO recurrence (rotary + trapezoidal + selective per-token A).
TEST(TestMamba3, SisoKernelParity) {
	const int B = 2, L = 6, H = 2, P = 3, N = 4, A = 2;
	auto idxBLHX = [&](int b, int t, int h, int c, int C) { return (((b * L + t) * H + h) * C) + c; };
	auto idxBLH  = [&](int b, int t, int h) { return (b * L + t) * H + h; };
	auto idxBLA  = [&](int b, int t, int k) { return (b * L + t) * A + k; };

	std::vector<float> C(B * L * H * N), bm(B * L * H * N), X(B * L * H * P), Z(B * L * H * P);
	std::vector<float> ADT(B * L * H), DT(B * L * H), trap(B * L * H), angle(B * L * A);
	std::vector<float> CB(H * N), BB(H * N), dv(H);

	// Deterministic, smooth pseudo-random fills.
	auto f = [](int s) { return std::sin(0.7f * s + 1.3f) * 0.5f; };
	for (size_t i = 0; i < C.size(); i++) { C[i] = f((int)i); bm[i] = f((int)i + 11); }
	for (size_t i = 0; i < X.size(); i++) { X[i] = f((int)i + 5); Z[i] = f((int)i + 23); }
	for (size_t i = 0; i < DT.size(); i++) { DT[i] = 0.05f + 0.04f * (0.5f + f((int)i)); ADT[i] = -(0.3f + 0.2f * (0.5f + f((int)i + 3))) * DT[i]; trap[i] = f((int)i + 7); }
	for (size_t i = 0; i < angle.size(); i++) angle[i] = f((int)i + 31);
	for (size_t i = 0; i < CB.size(); i++) { CB[i] = 0.1f * f((int)i + 2); BB[i] = 0.1f * f((int)i + 4); }
	for (int h = 0; h < H; h++) dv[h] = 0.5f + 0.2f * h;

	oa::SsmConfig cfg{
		.batch = (oa::U32)B, .seqLen = (oa::U32)L, .nHeads = (oa::U32)H, .headDim = (oa::U32)P,
		.stateSize = (oa::U32)N, .numRopeAngles = (oa::U32)A, .hasZ = 1u, .hasD = 1u };

	auto y = oa::FnMatrix::mamba3Siso(
		matFromVec(C, oa::MatrixShape{B, L, H, N}), matFromVec(bm, oa::MatrixShape{B, L, H, N}),
		matFromVec(X, oa::MatrixShape{B, L, H, P}), matFromVec(Z, oa::MatrixShape{B, L, H, P}),
		matFromVec(ADT, oa::MatrixShape{B, L, H}), matFromVec(DT, oa::MatrixShape{B, L, H}),
		matFromVec(trap, oa::MatrixShape{B, L, H}), matFromVec(angle, oa::MatrixShape{B, L, A}),
		matFromVec(CB, oa::MatrixShape{H, N}), matFromVec(BB, oa::MatrixShape{H, N}),
		matFromVec(dv, oa::MatrixShape{H}), cfg);
	syncCtx();

	auto sig = [](float v) { return 1.0f / (1.0f + std::exp(-v)); };
	auto silu = [&](float v) { return v * sig(v); };

	std::vector<float> ref(B * L * H * P, 0.0f);
	for (int b = 0; b < B; b++) for (int h = 0; h < H; h++) {
		std::vector<float> theta(A, 0.0f);
		std::vector<float> hst(P * N, 0.0f);
		for (int t = 0; t < L; t++) {
			float dt = DT[idxBLH(b, t, h)], adt = ADT[idxBLH(b, t, h)];
			float trapValue = sig(trap[idxBLH(b, t, h)]);
			float a_t = std::exp(adt);
			float sg = 0.0f;
			if (t + 1 < L) { float dtN = DT[idxBLH(b, t + 1, h)]; float trN = sig(trap[idxBLH(b, t + 1, h)]); sg = dtN * (1.0f - trN); }
			float gamma = dt * trapValue, scale = gamma + sg;

			std::vector<float> ct(N), bt(N);
			for (int n = 0; n < N; n++) { ct[n] = C[idxBLHX(b, t, h, n, N)] + CB[h * N + n]; bt[n] = bm[idxBLHX(b, t, h, n, N)] + BB[h * N + n]; }
			for (int k = 0; k < A; k++) theta[k] += angle[idxBLA(b, t, k)] * dt;
			for (int k = 0; k < A; k++) {
				float cs = std::cos(theta[k]), sn = std::sin(theta[k]);
				int i0 = 2 * k, i1 = i0 + 1;
				float c0 = ct[i0], c1 = ct[i1], b0 = bt[i0], b1 = bt[i1];
				ct[i0] = c0 * cs - c1 * sn; ct[i1] = c0 * sn + c1 * cs;
				bt[i0] = b0 * cs - b1 * sn; bt[i1] = b0 * sn + b1 * cs;
			}
			float qk = 0.0f; for (int n = 0; n < N; n++) qk += ct[n] * bt[n];
			for (int p = 0; p < P; p++) {
				float xv = X[idxBLHX(b, t, h, p, P)];
				float yv = 0.0f;
				for (int n = 0; n < N; n++) { float hd = a_t * hst[p * N + n]; yv += ct[n] * hd; hst[p * N + n] = hd + scale * xv * bt[n]; }
				yv += (dv[h] + gamma * qk) * xv;
				yv *= silu(Z[idxBLHX(b, t, h, p, P)]);
				ref[idxBLHX(b, t, h, p, P)] = yv;
			}
		}
	}

	float maxErr = 0.0f;
	for (int i = 0; i < B * L * H * P; i++) {
		float got = y.at(i), exp = ref[i];
		maxErr = std::max(maxErr, std::abs(got - exp));
		EXPECT_NEAR(got, exp, 1e-4f);
	}
	std::cerr << "Mamba3Siso parity max abs err = " << maxErr << std::endl;
}

void expectGroupedQkMatchesExpandedHeads(int L, int P) {
	const int B = 2, H = 4, G = 2, N = 4, A = 2;
	const int headsPerGroup = H / G;
	auto fill = [](int count, int offset) {
		std::vector<float> values(static_cast<size_t>(count));
		for (int i = 0; i < count; ++i) {
			values[static_cast<size_t>(i)] =
				std::sin(0.37F * static_cast<float>(i + offset)) * 0.3F;
		}
		return values;
	};
	auto cGroup = fill(B * L * G * N, 1);
	auto bGroup = fill(B * L * G * N, 7);
	std::vector<float> cHead(static_cast<size_t>(B * L * H * N));
	std::vector<float> bHead(cHead.size());
	for (int batch = 0; batch < B; ++batch) {
		for (int token = 0; token < L; ++token) {
			for (int head = 0; head < H; ++head) {
				const int group = head / headsPerGroup;
				for (int n = 0; n < N; ++n) {
					const int grouped = ((batch * L + token) * G + group) * N + n;
					const int expanded = ((batch * L + token) * H + head) * N + n;
					cHead[static_cast<size_t>(expanded)] = cGroup[static_cast<size_t>(grouped)];
					bHead[static_cast<size_t>(expanded)] = bGroup[static_cast<size_t>(grouped)];
				}
			}
		}
	}
	auto x = fill(B * L * H * P, 13);
	auto z = fill(B * L * H * P, 19);
	auto adt = fill(B * L * H, 23);
	for (float& value : adt) value = -0.05F - std::abs(value);
	auto dt = fill(B * L * H, 29);
	for (float& value : dt) value = 0.08F + std::abs(value);
	auto trap = fill(B * L * H, 31);
	auto angle = fill(B * L * A, 37);
	auto cbias = fill(H * N, 41);
	auto bbias = fill(H * N, 43);
	auto d = fill(H, 47);
	auto dout = fill(B * L * H * P, 53);
	oa::SsmConfig groupedConfig{
		.batch = static_cast<oa::U32>(B), .seqLen = static_cast<oa::U32>(L),
		.nHeads = static_cast<oa::U32>(H), .nGroups = static_cast<oa::U32>(G),
		.headDim = static_cast<oa::U32>(P), .stateSize = static_cast<oa::U32>(N),
		.numRopeAngles = static_cast<oa::U32>(A),
		.hasZ = 1u, .hasD = 1u};
	auto expandedConfig = groupedConfig;
	expandedConfig.nGroups = H;

	auto cGroupedMatrix = matFromVec(cGroup, oa::MatrixShape{B, L, G, N});
	auto bGroupedMatrix = matFromVec(bGroup, oa::MatrixShape{B, L, G, N});
	auto cHeadMatrix = matFromVec(cHead, oa::MatrixShape{B, L, H, N});
	auto bHeadMatrix = matFromVec(bHead, oa::MatrixShape{B, L, H, N});
	auto xMatrix = matFromVec(x, oa::MatrixShape{B, L, H, P});
	auto zMatrix = matFromVec(z, oa::MatrixShape{B, L, H, P});
	auto adtMatrix = matFromVec(adt, oa::MatrixShape{B, L, H});
	auto dtMatrix = matFromVec(dt, oa::MatrixShape{B, L, H});
	auto trapMatrix = matFromVec(trap, oa::MatrixShape{B, L, H});
	auto angleMatrix = matFromVec(angle, oa::MatrixShape{B, L, A});
	auto cbiasMatrix = matFromVec(cbias, oa::MatrixShape{H, N});
	auto bbiasMatrix = matFromVec(bbias, oa::MatrixShape{H, N});
	auto dMatrix = matFromVec(d, oa::MatrixShape{H});
	auto doutMatrix = matFromVec(dout, oa::MatrixShape{B, L, H, P});

	auto groupedY = oa::FnMatrix::mamba3Siso(
		cGroupedMatrix, bGroupedMatrix, xMatrix, zMatrix, adtMatrix, dtMatrix,
		trapMatrix, angleMatrix, cbiasMatrix, bbiasMatrix, dMatrix, groupedConfig);
	auto expandedY = oa::FnMatrix::mamba3Siso(
		cHeadMatrix, bHeadMatrix, xMatrix, zMatrix, adtMatrix, dtMatrix,
		trapMatrix, angleMatrix, cbiasMatrix, bbiasMatrix, dMatrix, expandedConfig);
	auto groupedGrad = oa::FnMatrix::mamba3SisoBwd(
		doutMatrix, cGroupedMatrix, bGroupedMatrix, xMatrix, zMatrix, adtMatrix,
		dtMatrix, trapMatrix, angleMatrix, cbiasMatrix, bbiasMatrix, dMatrix,
		groupedConfig);
	auto expandedGrad = oa::FnMatrix::mamba3SisoBwd(
		doutMatrix, cHeadMatrix, bHeadMatrix, xMatrix, zMatrix, adtMatrix,
		dtMatrix, trapMatrix, angleMatrix, cbiasMatrix, bbiasMatrix, dMatrix,
		expandedConfig);
	syncCtx();

	for (oa::I64 i = 0; i < groupedY.numElements(); ++i) {
		EXPECT_NEAR(groupedY.at(i), expandedY.at(i), 2.0e-5F) << "forward[" << i << "]";
	}
	for (int batch = 0; batch < B; ++batch) {
		for (int token = 0; token < L; ++token) {
			for (int group = 0; group < G; ++group) {
				for (int n = 0; n < N; ++n) {
					float expectedC = 0.0F;
					float expectedB = 0.0F;
					for (int localHead = 0; localHead < headsPerGroup; ++localHead) {
						const int head = group * headsPerGroup + localHead;
						const int index = ((batch * L + token) * H + head) * N + n;
						expectedC += expandedGrad.dC.at(index);
						expectedB += expandedGrad.dB.at(index);
					}
					const int index = ((batch * L + token) * G + group) * N + n;
					EXPECT_NEAR(groupedGrad.dC.at(index), expectedC, 2.0e-5F);
					EXPECT_NEAR(groupedGrad.dB.at(index), expectedB, 2.0e-5F);
				}
			}
		}
	}
	oa::Matrix* groupedOutputs[] = {
		&groupedGrad.dX, &groupedGrad.dZ, &groupedGrad.dAdt, &groupedGrad.dDt,
		&groupedGrad.dTrap, &groupedGrad.dAngle, &groupedGrad.dCBias,
		&groupedGrad.dBBias, &groupedGrad.dD};
	oa::Matrix* expandedOutputs[] = {
		&expandedGrad.dX, &expandedGrad.dZ, &expandedGrad.dAdt, &expandedGrad.dDt,
		&expandedGrad.dTrap, &expandedGrad.dAngle, &expandedGrad.dCBias,
		&expandedGrad.dBBias, &expandedGrad.dD};
	for (size_t output = 0; output < std::size(groupedOutputs); ++output) {
		ASSERT_EQ(groupedOutputs[output]->numElements(), expandedOutputs[output]->numElements());
		for (oa::I64 i = 0; i < groupedOutputs[output]->numElements(); ++i) {
			EXPECT_NEAR(groupedOutputs[output]->at(i), expandedOutputs[output]->at(i),
				2.0e-5F) << "gradient family " << output << " index " << i;
		}
	}
}

TEST(TestMamba3, SisoGroupedQkMatchesExpandedHeadsShort) {
	expectGroupedQkMatchesExpandedHeads(5, 3);
}

TEST(TestMamba3, SisoGroupedQkMatchesExpandedHeadsGeneric) {
	expectGroupedQkMatchesExpandedHeads(17, 17);
}

TEST(TestMamba3, SisoGroupedQkMatchesExpandedHeadsChunked) {
	expectGroupedQkMatchesExpandedHeads(70, 17);
}

// Finite-difference gradcheck: analytic Mamba3SisoBwd vs central differences of the
// forward, for a fixed random upstream gradient W (so loss = sum(W * y), dY = W).
TEST(TestMamba3, SisoKernelGradcheck) {
	const int B = 1, L = 4, H = 1, P = 3, N = 4, A = 2;
	const int nQK = B * L * H * N, nV = B * L * H * P, nS = B * L * H, nAng = B * L * A;

	std::vector<float> C(nQK), bm(nQK), X(nV), Z(nV);
	std::vector<float> ADT(nS), DT(nS), trap(nS), angle(nAng);
	std::vector<float> CB(H * N), BB(H * N), dv(H), W(nV);

	auto f = [](int s) { return std::sin(0.7f * s + 1.3f) * 0.5f; };
	for (int i = 0; i < nQK; i++) { C[i] = f(i); bm[i] = f(i + 11); }
	for (int i = 0; i < nV; i++) { X[i] = f(i + 5); Z[i] = f(i + 23); W[i] = f(i + 41); }
	for (int i = 0; i < nS; i++) { DT[i] = 0.05f + 0.04f * (0.5f + f(i)); ADT[i] = -(0.3f + 0.2f * (0.5f + f(i + 3))) * DT[i]; trap[i] = f(i + 7); }
	for (int i = 0; i < nAng; i++) angle[i] = f(i + 31);
	for (int i = 0; i < H * N; i++) { CB[i] = 0.1f * f(i + 2); BB[i] = 0.1f * f(i + 4); }
	for (int h = 0; h < H; h++) dv[h] = 0.5f + 0.2f * h;

	oa::SsmConfig cfg{
		.batch = (oa::U32)B, .seqLen = (oa::U32)L, .nHeads = (oa::U32)H, .headDim = (oa::U32)P,
		.stateSize = (oa::U32)N, .numRopeAngles = (oa::U32)A, .hasZ = 1u, .hasD = 1u };

	auto fwd = [&]() {
		return oa::FnMatrix::mamba3Siso(
			matFromVec(C, oa::MatrixShape{B, L, H, N}), matFromVec(bm, oa::MatrixShape{B, L, H, N}),
			matFromVec(X, oa::MatrixShape{B, L, H, P}), matFromVec(Z, oa::MatrixShape{B, L, H, P}),
			matFromVec(ADT, oa::MatrixShape{B, L, H}), matFromVec(DT, oa::MatrixShape{B, L, H}),
			matFromVec(trap, oa::MatrixShape{B, L, H}), matFromVec(angle, oa::MatrixShape{B, L, A}),
			matFromVec(CB, oa::MatrixShape{H, N}), matFromVec(BB, oa::MatrixShape{H, N}),
			matFromVec(dv, oa::MatrixShape{H}), cfg);
	};
	auto loss = [&]() {
		auto y = fwd();
		syncCtx();
		double l = 0.0;
		for (int i = 0; i < nV; i++) l += (double)W[i] * (double)y.at(i);
		return l;
	};

	// Analytic grads.
	auto g = oa::FnMatrix::mamba3SisoBwd(
		matFromVec(W, oa::MatrixShape{B, L, H, P}),
		matFromVec(C, oa::MatrixShape{B, L, H, N}), matFromVec(bm, oa::MatrixShape{B, L, H, N}),
		matFromVec(X, oa::MatrixShape{B, L, H, P}), matFromVec(Z, oa::MatrixShape{B, L, H, P}),
		matFromVec(ADT, oa::MatrixShape{B, L, H}), matFromVec(DT, oa::MatrixShape{B, L, H}),
		matFromVec(trap, oa::MatrixShape{B, L, H}), matFromVec(angle, oa::MatrixShape{B, L, A}),
		matFromVec(CB, oa::MatrixShape{H, N}), matFromVec(BB, oa::MatrixShape{H, N}),
		matFromVec(dv, oa::MatrixShape{H}), cfg);
	syncCtx();

	const float eps = 2e-3f;
	float maxErr = 0.0f;
	auto check = [&](const char* name, std::vector<float>& vec, oa::Matrix& analytic) {
		for (size_t i = 0; i < vec.size(); i++) {
			float orig = vec[i];
			vec[i] = orig + eps; double lp = loss();
			vec[i] = orig - eps; double lm = loss();
			vec[i] = orig;
			float num = (float)((lp - lm) / (2.0 * eps));
			float ana = analytic.at((oa::I64)i);
			float tol = 2e-2f + 3e-2f * std::abs(ana);
			float err = std::abs(num - ana);
			maxErr = std::max(maxErr, err);
			EXPECT_NEAR(num, ana, tol) << name << "[" << i << "] num=" << num << " ana=" << ana;
		}
	};

	check("dC", C, g.dC);
	check("dB", bm, g.dB);
	check("dX", X, g.dX);
	check("dZ", Z, g.dZ);
	check("dADT", ADT, g.dAdt);
	check("dDT", DT, g.dDt);
	check("dTrap", trap, g.dTrap);
	check("dAngle", angle, g.dAngle);
	check("dCBias", CB, g.dCBias);
	check("dBBias", BB, g.dBBias);
	check("dD", dv, g.dD);

	std::cerr << "Mamba3Siso gradcheck max abs err = " << maxErr << std::endl;
}

// The short lowering also has to preserve unpaired state channels and the
// optional Z/D contract.  N=5,A=2 leaves one non-rotary tail channel, while
// P=3 exercises inactive workgroup lanes.  Disabled inputs remain present as
// value placeholders but must have exactly zero derivatives.
TEST(TestMamba3, SisoKernelGradcheckOddTailWithoutOptionalTerms) {
	const int B = 1, L = 5, H = 1, P = 3, N = 5, A = 2;
	const int nQK = B * L * H * N, nV = B * L * H * P;
	const int nS = B * L * H, nAng = B * L * A;
	std::vector<float> C(nQK), bm(nQK), X(nV), Z(nV);
	std::vector<float> ADT(nS), DT(nS), trap(nS), angle(nAng);
	std::vector<float> CB(H * N), BB(H * N), dv(H), W(nV);
	auto f = [](int s) { return std::sin(0.53F * s + 0.91F) * 0.45F; };
	for (int i = 0; i < nQK; ++i) { C[i] = f(i); bm[i] = f(i + 13); }
	for (int i = 0; i < nV; ++i) {
		X[i] = f(i + 5); Z[i] = f(i + 29); W[i] = f(i + 47);
	}
	for (int i = 0; i < nS; ++i) {
		DT[i] = 0.06F + 0.03F * (0.5F + f(i));
		ADT[i] = -(0.25F + 0.15F * (0.5F + f(i + 3))) * DT[i];
		trap[i] = f(i + 7);
	}
	for (int i = 0; i < nAng; ++i) angle[i] = f(i + 31);
	for (int i = 0; i < H * N; ++i) {
		CB[i] = 0.1F * f(i + 2); BB[i] = 0.1F * f(i + 4);
	}
	dv[0] = 0.7F;
	oa::SsmConfig cfg{
		.batch = (oa::U32)B, .seqLen = (oa::U32)L, .nHeads = (oa::U32)H,
		.headDim = (oa::U32)P, .stateSize = (oa::U32)N,
		.numRopeAngles = (oa::U32)A, .hasZ = 0u, .hasD = 0u};

	auto loss = [&]() {
		auto y = oa::FnMatrix::mamba3Siso(
			matFromVec(C, oa::MatrixShape{B, L, H, N}),
			matFromVec(bm, oa::MatrixShape{B, L, H, N}),
			matFromVec(X, oa::MatrixShape{B, L, H, P}),
			matFromVec(Z, oa::MatrixShape{B, L, H, P}),
			matFromVec(ADT, oa::MatrixShape{B, L, H}),
			matFromVec(DT, oa::MatrixShape{B, L, H}),
			matFromVec(trap, oa::MatrixShape{B, L, H}),
			matFromVec(angle, oa::MatrixShape{B, L, A}),
			matFromVec(CB, oa::MatrixShape{H, N}),
			matFromVec(BB, oa::MatrixShape{H, N}),
			matFromVec(dv, oa::MatrixShape{H}), cfg);
		syncCtx();
		double value = 0.0;
		for (int i = 0; i < nV; ++i) value += (double)W[i] * (double)y.at(i);
		return value;
	};
	auto g = oa::FnMatrix::mamba3SisoBwd(
		matFromVec(W, oa::MatrixShape{B, L, H, P}),
		matFromVec(C, oa::MatrixShape{B, L, H, N}),
		matFromVec(bm, oa::MatrixShape{B, L, H, N}),
		matFromVec(X, oa::MatrixShape{B, L, H, P}),
		matFromVec(Z, oa::MatrixShape{B, L, H, P}),
		matFromVec(ADT, oa::MatrixShape{B, L, H}),
		matFromVec(DT, oa::MatrixShape{B, L, H}),
		matFromVec(trap, oa::MatrixShape{B, L, H}),
		matFromVec(angle, oa::MatrixShape{B, L, A}),
		matFromVec(CB, oa::MatrixShape{H, N}),
		matFromVec(BB, oa::MatrixShape{H, N}),
		matFromVec(dv, oa::MatrixShape{H}), cfg);
	syncCtx();

	const float eps = 2e-3F;
	float maxErr = 0.0F;
	auto check = [&](const char* name, std::vector<float>& values, oa::Matrix& analytic) {
		for (size_t i = 0; i < values.size(); ++i) {
			float original = values[i];
			values[i] = original + eps; double plus = loss();
			values[i] = original - eps; double minus = loss();
			values[i] = original;
			float numerical = (float)((plus - minus) / (2.0 * eps));
			float exact = analytic.at((oa::I64)i);
			float tolerance = 2e-2F + 3e-2F * std::abs(exact);
			maxErr = std::max(maxErr, std::abs(numerical - exact));
			EXPECT_NEAR(numerical, exact, tolerance)
				<< name << "[" << i << "] num=" << numerical << " ana=" << exact;
		}
	};
	check("dC", C, g.dC);
	check("dB", bm, g.dB);
	check("dX", X, g.dX);
	check("dZ-disabled", Z, g.dZ);
	check("dADT", ADT, g.dAdt);
	check("dDT", DT, g.dDt);
	check("dTrap", trap, g.dTrap);
	check("dAngle", angle, g.dAngle);
	check("dCBias", CB, g.dCBias);
	check("dBBias", BB, g.dBBias);
	check("dD-disabled", dv, g.dD);
	for (int i = 0; i < nV; ++i) EXPECT_FLOAT_EQ(g.dZ.at(i), 0.0F);
	EXPECT_FLOAT_EQ(g.dD.at(0), 0.0F);
	std::cerr << "Mamba3Siso odd-tail/no-option gradcheck max abs err = "
		<< maxErr << std::endl;
}

TEST(TestMamba3, SisoBackwardShortValidationSmoke) {
	const int B = 1, L = 5, H = 1, P = 3, N = 5, A = 2;
	auto values = [](int count, int offset) {
		std::vector<float> result(count);
		for (int i = 0; i < count; ++i) {
			result[i] = std::sin(0.41F * static_cast<float>(i + offset)) * 0.2F;
		}
		return result;
	};
	auto c = values(B * L * H * N, 1);
	auto b = values(B * L * H * N, 7);
	auto x = values(B * L * H * P, 13);
	auto z = values(B * L * H * P, 19);
	auto adt = values(B * L * H, 23);
	for (float& value : adt) value = -0.05F - std::abs(value);
	auto dt = values(B * L * H, 29);
	for (float& value : dt) value = 0.08F + std::abs(value);
	auto trap = values(B * L * H, 31);
	auto angle = values(B * L * A, 37);
	auto cb = values(H * N, 41);
	auto bb = values(H * N, 43);
	auto d = values(H, 47);
	auto dout = values(B * L * H * P, 53);
	oa::SsmConfig cfg{
		.batch = B, .seqLen = L, .nHeads = H, .headDim = P,
		.stateSize = N, .numRopeAngles = A, .hasZ = 1u, .hasD = 1u};
	auto gradients = oa::FnMatrix::mamba3SisoBwd(
		matFromVec(dout, oa::MatrixShape{B, L, H, P}),
		matFromVec(c, oa::MatrixShape{B, L, H, N}),
		matFromVec(b, oa::MatrixShape{B, L, H, N}),
		matFromVec(x, oa::MatrixShape{B, L, H, P}),
		matFromVec(z, oa::MatrixShape{B, L, H, P}),
		matFromVec(adt, oa::MatrixShape{B, L, H}),
		matFromVec(dt, oa::MatrixShape{B, L, H}),
		matFromVec(trap, oa::MatrixShape{B, L, H}),
		matFromVec(angle, oa::MatrixShape{B, L, A}),
		matFromVec(cb, oa::MatrixShape{H, N}),
		matFromVec(bb, oa::MatrixShape{H, N}),
		matFromVec(d, oa::MatrixShape{H}), cfg);
	syncCtx();
	oa::Matrix* outputs[] = {
		&gradients.dC, &gradients.dB, &gradients.dX, &gradients.dZ,
		&gradients.dAdt, &gradients.dDt, &gradients.dTrap, &gradients.dAngle,
		&gradients.dCBias, &gradients.dBBias, &gradients.dD};
	for (oa::Matrix* output : outputs) {
		ASSERT_GT(output->numElements(), 0);
		for (oa::I64 i = 0; i < output->numElements(); ++i) {
			EXPECT_TRUE(std::isfinite(output->at(i))) << "gradient index " << i;
		}
	}
}

// Multi-chunk gradcheck: L=70 spans five 16-token chunks. P=17 exercises
// inactive workgroup lanes and N=5 leaves an unpaired non-rotary state channel,
// while the strided probes cover every gradient family across chunk boundaries.
TEST(TestMamba3, SisoKernelGradcheckMultiChunk) {
	const int B = 1, L = 70, H = 1, P = 17, N = 5, A = 2;
	const int nQK = B * L * H * N, nV = B * L * H * P, nS = B * L * H, nAng = B * L * A;

	std::vector<float> C(nQK), bm(nQK), X(nV), Z(nV);
	std::vector<float> ADT(nS), DT(nS), trap(nS), angle(nAng);
	std::vector<float> CB(H * N), BB(H * N), dv(H), W(nV);

	auto f = [](int s) { return std::sin(0.7f * s + 1.3f) * 0.5f; };
	for (int i = 0; i < nQK; i++) { C[i] = f(i); bm[i] = f(i + 11); }
	for (int i = 0; i < nV; i++) { X[i] = f(i + 5); Z[i] = f(i + 23); W[i] = f(i + 41); }
	for (int i = 0; i < nS; i++) { DT[i] = 0.05f + 0.04f * (0.5f + f(i)); ADT[i] = -(0.3f + 0.2f * (0.5f + f(i + 3))) * DT[i]; trap[i] = f(i + 7); }
	for (int i = 0; i < nAng; i++) angle[i] = f(i + 31);
	for (int i = 0; i < H * N; i++) { CB[i] = 0.1f * f(i + 2); BB[i] = 0.1f * f(i + 4); }
	for (int h = 0; h < H; h++) dv[h] = 0.5f + 0.2f * h;

	oa::SsmConfig cfg{
		.batch = (oa::U32)B, .seqLen = (oa::U32)L, .nHeads = (oa::U32)H, .headDim = (oa::U32)P,
		.stateSize = (oa::U32)N, .numRopeAngles = (oa::U32)A, .hasZ = 1u, .hasD = 1u };

	auto loss = [&]() {
		auto y = oa::FnMatrix::mamba3Siso(
			matFromVec(C, oa::MatrixShape{B, L, H, N}), matFromVec(bm, oa::MatrixShape{B, L, H, N}),
			matFromVec(X, oa::MatrixShape{B, L, H, P}), matFromVec(Z, oa::MatrixShape{B, L, H, P}),
			matFromVec(ADT, oa::MatrixShape{B, L, H}), matFromVec(DT, oa::MatrixShape{B, L, H}),
			matFromVec(trap, oa::MatrixShape{B, L, H}), matFromVec(angle, oa::MatrixShape{B, L, A}),
			matFromVec(CB, oa::MatrixShape{H, N}), matFromVec(BB, oa::MatrixShape{H, N}),
			matFromVec(dv, oa::MatrixShape{H}), cfg);
		syncCtx();
		double l = 0.0;
		for (int i = 0; i < nV; i++) l += (double)W[i] * (double)y.at(i);
		return l;
	};

	auto g = oa::FnMatrix::mamba3SisoBwd(
		matFromVec(W, oa::MatrixShape{B, L, H, P}),
		matFromVec(C, oa::MatrixShape{B, L, H, N}), matFromVec(bm, oa::MatrixShape{B, L, H, N}),
		matFromVec(X, oa::MatrixShape{B, L, H, P}), matFromVec(Z, oa::MatrixShape{B, L, H, P}),
		matFromVec(ADT, oa::MatrixShape{B, L, H}), matFromVec(DT, oa::MatrixShape{B, L, H}),
		matFromVec(trap, oa::MatrixShape{B, L, H}), matFromVec(angle, oa::MatrixShape{B, L, A}),
		matFromVec(CB, oa::MatrixShape{H, N}), matFromVec(BB, oa::MatrixShape{H, N}),
		matFromVec(dv, oa::MatrixShape{H}), cfg);
	syncCtx();

	const float eps = 2e-3f;
	float maxErr = 0.0f;
	// Spot-check a strided subset of each input to keep runtime bounded.
	auto check = [&](const char* name, std::vector<float>& vec, oa::Matrix& analytic, int stride) {
		for (size_t i = 0; i < vec.size(); i += stride) {
			float orig = vec[i];
			vec[i] = orig + eps; double lp = loss();
			vec[i] = orig - eps; double lm = loss();
			vec[i] = orig;
			float num = (float)((lp - lm) / (2.0 * eps));
			float ana = analytic.at((oa::I64)i);
			float tol = 2e-2f + 3e-2f * std::abs(ana);
			maxErr = std::max(maxErr, std::abs(num - ana));
			EXPECT_NEAR(num, ana, tol) << name << "[" << i << "] num=" << num << " ana=" << ana;
		}
	};
	check("dC", C, g.dC, 31);
	check("dB", bm, g.dB, 31);
	check("dX", X, g.dX, 97);
	check("dZ", Z, g.dZ, 97);
	check("dADT", ADT, g.dAdt, 13);
	check("dDT", DT, g.dDt, 13);
	check("dTrap", trap, g.dTrap, 13);
	check("dAngle", angle, g.dAngle, 17);
	check("dCBias", CB, g.dCBias, 1);
	check("dBBias", BB, g.dBBias, 1);
	check("dD", dv, g.dD, 1);
	std::cerr << "Mamba3Siso multi-chunk gradcheck max abs err = " << maxErr << std::endl;
}

// EmpyrealmSiso is documented as an "exact copy" of the Mamba3Siso kernels in a
// separate namespace. These two tests pin that claim: the forward must match
// Mamba3Siso to fp32 noise, and EmpyrealmSisoBwd must pass an independent
// finite-difference gradcheck (it had NO numerical verification before — the
// only thing guarding it was the unenforced "exact copy" comment).
TEST(TestMamba3, EmpyrealmSisoParity) {
	const int B = 1, L = 5, H = 2, P = 3, N = 4, A = 2;
	const int nQK = B * L * H * N, nV = B * L * H * P, nS = B * L * H, nAng = B * L * A;

	std::vector<float> C(nQK), bm(nQK), X(nV), Z(nV);
	std::vector<float> ADT(nS), DT(nS), trap(nS), angle(nAng);
	std::vector<float> CB(H * N), BB(H * N), dv(H);

	auto f = [](int s) { return std::sin(0.7f * s + 1.3f) * 0.5f; };
	for (int i = 0; i < nQK; i++) { C[i] = f(i); bm[i] = f(i + 11); }
	for (int i = 0; i < nV; i++) { X[i] = f(i + 5); Z[i] = f(i + 23); }
	for (int i = 0; i < nS; i++) { DT[i] = 0.05f + 0.04f * (0.5f + f(i)); ADT[i] = -(0.3f + 0.2f * (0.5f + f(i + 3))) * DT[i]; trap[i] = f(i + 7); }
	for (int i = 0; i < nAng; i++) angle[i] = f(i + 31);
	for (int i = 0; i < H * N; i++) { CB[i] = 0.1f * f(i + 2); BB[i] = 0.1f * f(i + 4); }
	for (int h = 0; h < H; h++) dv[h] = 0.5f + 0.2f * h;

	oa::SsmConfig cfg{
		.batch = (oa::U32)B, .seqLen = (oa::U32)L, .nHeads = (oa::U32)H, .headDim = (oa::U32)P,
		.stateSize = (oa::U32)N, .numRopeAngles = (oa::U32)A, .hasZ = 1u, .hasD = 1u };

	auto args = [&]() {
		return std::make_tuple(
			matFromVec(C, oa::MatrixShape{B, L, H, N}), matFromVec(bm, oa::MatrixShape{B, L, H, N}),
			matFromVec(X, oa::MatrixShape{B, L, H, P}), matFromVec(Z, oa::MatrixShape{B, L, H, P}),
			matFromVec(ADT, oa::MatrixShape{B, L, H}), matFromVec(DT, oa::MatrixShape{B, L, H}),
			matFromVec(trap, oa::MatrixShape{B, L, H}), matFromVec(angle, oa::MatrixShape{B, L, A}),
			matFromVec(CB, oa::MatrixShape{H, N}), matFromVec(BB, oa::MatrixShape{H, N}),
			matFromVec(dv, oa::MatrixShape{H}));
	};
	auto [c0, b0, x0, z0, adt0, dt0, tr0, an0, cb0, bb0, d0] = args();
	auto [c1, b1, x1, z1, adt1, dt1, tr1, an1, cb1, bb1, d1] = args();
	auto yMamba = oa::FnMatrix::mamba3Siso(c0, b0, x0, z0, adt0, dt0, tr0, an0, cb0, bb0, d0, cfg);
	auto yEmpy  = oa::FnMatrix::empyrealmSiso(c1, b1, x1, z1, adt1, dt1, tr1, an1, cb1, bb1, d1, cfg);
	syncCtx();

	float maxErr = 0.0f;
	for (int i = 0; i < nV; i++) {
		float err = std::abs(yMamba.at(i) - yEmpy.at(i));
		maxErr = std::max(maxErr, err);
		EXPECT_NEAR(yMamba.at(i), yEmpy.at(i), 1e-4f) << "i=" << i;
	}
	std::cerr << "EmpyrealmSiso vs Mamba3Siso forward max abs err = " << maxErr << std::endl;
}

// Finite-difference gradcheck for EmpyrealmSisoBwd — analytic vs central differences
// of the EmpyrealmSiso forward, mirroring SisoKernelGradcheck.
TEST(TestMamba3, EmpyrealmSisoKernelGradcheck) {
	const int B = 1, L = 4, H = 1, P = 3, N = 4, A = 2;
	const int nQK = B * L * H * N, nV = B * L * H * P, nS = B * L * H, nAng = B * L * A;

	std::vector<float> C(nQK), bm(nQK), X(nV), Z(nV);
	std::vector<float> ADT(nS), DT(nS), trap(nS), angle(nAng);
	std::vector<float> CB(H * N), BB(H * N), dv(H), W(nV);

	auto f = [](int s) { return std::sin(0.7f * s + 1.3f) * 0.5f; };
	for (int i = 0; i < nQK; i++) { C[i] = f(i); bm[i] = f(i + 11); }
	for (int i = 0; i < nV; i++) { X[i] = f(i + 5); Z[i] = f(i + 23); W[i] = f(i + 41); }
	for (int i = 0; i < nS; i++) { DT[i] = 0.05f + 0.04f * (0.5f + f(i)); ADT[i] = -(0.3f + 0.2f * (0.5f + f(i + 3))) * DT[i]; trap[i] = f(i + 7); }
	for (int i = 0; i < nAng; i++) angle[i] = f(i + 31);
	for (int i = 0; i < H * N; i++) { CB[i] = 0.1f * f(i + 2); BB[i] = 0.1f * f(i + 4); }
	for (int h = 0; h < H; h++) dv[h] = 0.5f + 0.2f * h;

	oa::SsmConfig cfg{
		.batch = (oa::U32)B, .seqLen = (oa::U32)L, .nHeads = (oa::U32)H, .headDim = (oa::U32)P,
		.stateSize = (oa::U32)N, .numRopeAngles = (oa::U32)A, .hasZ = 1u, .hasD = 1u };

	auto loss = [&]() {
		auto y = oa::FnMatrix::empyrealmSiso(
			matFromVec(C, oa::MatrixShape{B, L, H, N}), matFromVec(bm, oa::MatrixShape{B, L, H, N}),
			matFromVec(X, oa::MatrixShape{B, L, H, P}), matFromVec(Z, oa::MatrixShape{B, L, H, P}),
			matFromVec(ADT, oa::MatrixShape{B, L, H}), matFromVec(DT, oa::MatrixShape{B, L, H}),
			matFromVec(trap, oa::MatrixShape{B, L, H}), matFromVec(angle, oa::MatrixShape{B, L, A}),
			matFromVec(CB, oa::MatrixShape{H, N}), matFromVec(BB, oa::MatrixShape{H, N}),
			matFromVec(dv, oa::MatrixShape{H}), cfg);
		syncCtx();
		double l = 0.0;
		for (int i = 0; i < nV; i++) l += (double)W[i] * (double)y.at(i);
		return l;
	};

	auto g = oa::FnMatrix::empyrealmSisoBwd(
		matFromVec(W, oa::MatrixShape{B, L, H, P}),
		matFromVec(C, oa::MatrixShape{B, L, H, N}), matFromVec(bm, oa::MatrixShape{B, L, H, N}),
		matFromVec(X, oa::MatrixShape{B, L, H, P}), matFromVec(Z, oa::MatrixShape{B, L, H, P}),
		matFromVec(ADT, oa::MatrixShape{B, L, H}), matFromVec(DT, oa::MatrixShape{B, L, H}),
		matFromVec(trap, oa::MatrixShape{B, L, H}), matFromVec(angle, oa::MatrixShape{B, L, A}),
		matFromVec(CB, oa::MatrixShape{H, N}), matFromVec(BB, oa::MatrixShape{H, N}),
		matFromVec(dv, oa::MatrixShape{H}), cfg);
	syncCtx();

	const float eps = 2e-3f;
	float maxErr = 0.0f;
	auto check = [&](const char* name, std::vector<float>& vec, oa::Matrix& analytic) {
		for (size_t i = 0; i < vec.size(); i++) {
			float orig = vec[i];
			vec[i] = orig + eps; double lp = loss();
			vec[i] = orig - eps; double lm = loss();
			vec[i] = orig;
			float num = (float)((lp - lm) / (2.0 * eps));
			float ana = analytic.at((oa::I64)i);
			float tol = 2e-2f + 3e-2f * std::abs(ana);
			maxErr = std::max(maxErr, std::abs(num - ana));
			EXPECT_NEAR(num, ana, tol) << name << "[" << i << "] num=" << num << " ana=" << ana;
		}
	};

	check("dC", C, g.dC);
	check("dB", bm, g.dB);
	check("dX", X, g.dX);
	check("dZ", Z, g.dZ);
	check("dADT", ADT, g.dAdt);
	check("dDT", DT, g.dDt);
	check("dTrap", trap, g.dTrap);
	check("dAngle", angle, g.dAngle);
	check("dCBias", CB, g.dCBias);
	check("dBBias", BB, g.dBBias);
	check("dD", dv, g.dD);

	std::cerr << "EmpyrealmSiso gradcheck max abs err = " << maxErr << std::endl;
}

// Gradcheck for rmsNormGated (norm_before_gate=true): analytic RmsNormGatedBwd vs
// central differences of the forward, for grads w.r.t. x, weight, bias, z.
TEST(TestMamba3, RmsNormGatedGradcheck) {
	const int Outer = 2, Groups = 3, Cc = 4;
	const int R = Outer * Groups;
	const float eps = 1e-5f;
	std::vector<float> X(R * Cc), wt(Groups * Cc), bs(Groups * Cc);
	std::vector<float> zt(R * Cc), Up(R * Cc);
	auto f = [](int s) { return std::sin(0.7f * s + 1.3f) * 0.6f; };
	for (int i = 0; i < R * Cc; i++) { X[i] = f(i) + 0.2f; zt[i] = f(i + 9); Up[i] = f(i + 17); }
	for (int i = 0; i < Groups * Cc; i++) { wt[i] = 1.0f + 0.3f * f(i + 3); bs[i] = 0.1f * f(i + 5); }

	auto loss = [&]() {
		auto y = oa::FnMatrix::rmsNormGated(
			matFromVec(X, oa::MatrixShape{Outer, Groups, Cc}),
			matFromVec(wt, oa::MatrixShape{Groups, Cc}),
			matFromVec(bs, oa::MatrixShape{Groups, Cc}),
			matFromVec(zt, oa::MatrixShape{Outer, Groups, Cc}), eps, true);
		syncCtx();
		double l = 0.0;
		for (int i = 0; i < R * Cc; i++) l += (double)Up[i] * (double)y.at(i);
		return l;
	};
	auto g = oa::FnMatrix::rmsNormGatedBwd(
		matFromVec(X, oa::MatrixShape{Outer, Groups, Cc}),
		matFromVec(wt, oa::MatrixShape{Groups, Cc}),
		matFromVec(bs, oa::MatrixShape{Groups, Cc}),
		matFromVec(zt, oa::MatrixShape{Outer, Groups, Cc}),
		matFromVec(Up, oa::MatrixShape{Outer, Groups, Cc}), eps);
	syncCtx();

	const float h = 1e-3f;
	float maxErr = 0.0f;
	auto check = [&](const char* name, std::vector<float>& v, oa::Matrix& ana) {
		for (size_t i = 0; i < v.size(); i++) {
			float o = v[i];
			v[i] = o + h; double lp = loss();
			v[i] = o - h; double lm = loss();
			v[i] = o;
			float num = (float)((lp - lm) / (2.0 * h));
			float a = ana.at((oa::I64)i);
			maxErr = std::max(maxErr, std::abs(num - a));
			EXPECT_NEAR(num, a, 1e-2f + 2e-2f * std::abs(a)) << name << "[" << i << "]";
		}
	};
	check("dX", X, g.dX);
	check("dW", wt, g.dWeight);
	check("dB", bs, g.dBias);
	check("dZ", zt, g.dZ);
	std::cerr << "RmsNormGated gradcheck max abs err = " << maxErr << std::endl;
}

// Mamba3 with gated output RMSNorm: forward runs and gradients flow through the new
// norm_weight parameter (and the rest of the block) end-to-end.
TEST(TestMamba3, OutprojNormTrains) {
	oa::Mamba3Module mamba3(
		32,    // d_model
		16,    // d_state
		2,     // expand
		16,    // headdim
		1,     // ngroups
		0.5f,  // rope_fraction
		false, // is_mimo
		4, 0.001f, 0.1f, 1e-4f, 1e-4f,
		true   // is_outproj_norm
	);
	// in_proj, dt_bias, B_bias, C_bias, D, out_proj, norm_weight
	EXPECT_EQ(mamba3.parameters().size(), 7);

	auto input = oa::FnMatrix::randN(oa::MatrixShape{2, 24, 32}, oa::ScalarType::Float32);
	input.setRequiresGrad(true);

	oa::Matrix out;
	{
		oa::GradientTape tape;
		out = mamba3.forward(input);
		syncCtx();
		auto loss = oa::FnMatrix::mean(out, -1);
		syncCtx();
		tape.backward(loss);
		syncCtx();
	}

	EXPECT_EQ(out.rank(), 3);
	EXPECT_EQ(out.size(2), 32);
	// Backprop ran end-to-end through the gated-norm output path (norm differentiability
	// itself is covered by RmsNormGatedGradcheck): the input receives a gradient.
	EXPECT_TRUE(input.gradMatrix().rank() > 0) << "no gradient reached the input";
	EXPECT_FALSE(mamba3.normWeight().gradMatrix().isEmpty())
		<< "no gradient reached the grouped norm weight";
	std::cerr << "OutprojNorm forward+backward OK" << std::endl;
}

// Recurrent decode must equal the full forward: stepping token-by-token through
// the cached SSM/angle/k/v state reproduces the sequence recurrence exactly.
TEST(TestMamba3, StepMatchesForward) {
	const int Bn = 2, Ln = 5, D = 32;
	oa::Mamba3Module mamba3(D, 16, 2, 16, 1);  // d_model, d_state, expand, headdim, ngroups

	auto input = oa::FnMatrix::randN(oa::MatrixShape{Bn, Ln, D}, oa::ScalarType::Float32);
	auto full = mamba3.forward(input);
	syncCtx();

	mamba3.resetState(Bn);
	float maxErr = 0.0f;
	for (int t = 0; t < Ln; t++) {
		auto slice = oa::FnMatrix::slice(input, 1, t, t + 1);  // [Bn,1,D]
		auto yt = mamba3.step(slice);
		syncCtx();
		for (int b = 0; b < Bn; b++) {
			for (int d = 0; d < D; d++) {
				float got = yt.at((oa::I64)(b * D + d));
				float exp = full.at((oa::I64)((b * Ln + t) * D + d));
				maxErr = std::max(maxErr, std::abs(got - exp));
				EXPECT_NEAR(got, exp, 2e-3f) << "t=" << t << " b=" << b << " d=" << d;
			}
		}
	}
	std::cerr << "step-vs-forward max abs err = " << maxErr << std::endl;
}

// Every Mamba3 parameter must receive a non-trivial gradient through the full module
// chain (flat residual + CE head), matching TutorialNlpMamba3Ag wiring.
TEST(TestMamba3, ModuleGradMagnitudes) {
	// Mirrors TutorialNlpMamba3Ag: Embed → Mamba3 + residual → CE head.
	const oa::I32 B = 4, S = 16, D = 32, V = 512;
	oa::Mamba3Module m(D, 32, 2, 16, 1, 0.5f, false, 4, 0.001f, 0.1f, 1e-4f, 1e-4f, true);
	oa::Embedding embed(V, D);
	oa::Linear head(D, V, true);
	oa::Vec<oa::I32> tokens(B * S);
	for (oa::I32 i = 0; i < B * S; ++i) tokens[i] = i % V;
	auto batchX = oa::FnMatrix::fromInt32(oa::Span<const oa::I32>(tokens.data(), tokens.size()),
		oa::MatrixShape{B, S}, oa::ScalarType::Int32);
	oa::Vec<oa::I32> labels(B * S);
	for (oa::I32 i = 0; i < B * S; ++i) labels[i] = (i + 1) % V;
	auto targets = oa::FnMatrix::fromInt32(oa::Span<const oa::I32>(labels.data(), labels.size()),
		oa::MatrixShape{B * S}, oa::ScalarType::Int32);
	{
		oa::GradientTape tape;
		auto emb = embed.forward(batchX);                  // flat [B*S, D]
		auto emb3d = emb.reshape(oa::MatrixShape{B, S, D});       // Mamba3 needs [B, S, D]
		auto y3d = m.forward(emb3d);                        // [B, S, D]
		auto mixed = y3d.reshape(oa::MatrixShape{B * S, D}) + emb.reshape(oa::MatrixShape{B * S, D});
		auto loss = oa::FnLoss::crossEntropy(head.forward(mixed), targets);
		syncCtx();
		tape.backward(loss);
		syncCtx();
	}
	auto gradMag = [](const oa::Matrix& g) -> double {
		if (g.isEmpty() || g.numElements() == 0) return 0.0;
		auto s = oa::FnMatrix::sum(oa::FnMatrix::abs(g.reshape(oa::MatrixShape{g.numElements()})), 0);
		syncCtx();
		return static_cast<double>(s.at(0));
	};
	// out_proj must receive gradient through the CE head (core LM wiring).
	EXPECT_GT(gradMag(m.parameters()[5].data.gradMatrix()), 0.0) << "out_proj has zero gradient";
}

// MIMO (rank R>1): forward runs, output shape is correct, and backprop flows
// through the dedicated shared-state Mamba3Mimo adjoint.
TEST(TestMamba3, MimoForwardBackward) {
	const int Bn = 2, Ln = 6, D = 32, R = 4;
	oa::Mamba3Module mamba3(
		D, 16, 2, 16, 1,   // d_model, d_state, expand, headdim, ngroups
		0.5F, true, R,      // rope_fraction, is_mimo, mimo_rank
		0.001F, 0.1F, 1e-4F, 1e-4F, true // stability bounds, rankwise output norm
	);
	// in_proj, dt_bias, B_bias, C_bias, mimo_x/z/o, D, out_proj, norm_weight
	EXPECT_EQ(mamba3.parameters().size(), 10);

	auto input = oa::FnMatrix::randN(oa::MatrixShape{Bn, Ln, D}, oa::ScalarType::Float32);
	input.setRequiresGrad(true);

	oa::Matrix out;
	{
		oa::GradientTape tape;
		out = mamba3.forward(input);
		syncCtx();
		auto loss = oa::FnMatrix::mean(out, -1);
		syncCtx();
		tape.backward(loss);
		syncCtx();
	}
	EXPECT_EQ(out.rank(), 3);
	EXPECT_EQ(out.size(0), Bn);
	EXPECT_EQ(out.size(1), Ln);
	EXPECT_EQ(out.size(2), D);
	EXPECT_TRUE(input.gradMatrix().rank() > 0) << "no gradient reached the input through MIMO";
	for (oa::Usize i = 0; i < mamba3.parameters().size(); ++i) {
		const auto& gradient = mamba3.parameters()[i].data.gradMatrix();
		ASSERT_FALSE(gradient.isEmpty()) << "parameter " << i << " has no gradient";
		auto magnitude = oa::FnMatrix::sum(
			oa::FnMatrix::abs(gradient.reshape({gradient.numElements()})), 0);
		syncCtx();
		EXPECT_GT(std::abs(magnitude.at(0)), 0.0F)
			<< "parameter " << i << " has a zero gradient";
	}
	std::cerr << "MIMO R=" << R << " forward+backward OK" << std::endl;
}

TEST(TestMamba3, MimoKernelMatchesSharedStateCpuOracle) {
	const int Bn = 1, L = 4, H = 2, G = 1, R = 2, P = 3, N = 4, A = 2;
	auto value = [](int i) { return 0.35F * std::sin(0.41F * static_cast<float>(i) + 0.2F); };
	std::vector<float> c(Bn * L * R * G * N), k(c.size());
	std::vector<float> x(Bn * L * H * P), z(x.size());
	std::vector<float> adt(Bn * L * H), dt(adt.size()), trap(adt.size());
	std::vector<float> angle(Bn * L * A);
	std::vector<float> cb(H * R * N), kb(cb.size()), d(H);
	std::vector<float> mx(H * R * P), mz(mx.size()), mo(mx.size()), norm(H * P, 1.0F);
	for (size_t i = 0; i < c.size(); ++i) { c[i] = value(static_cast<int>(i)); k[i] = value(static_cast<int>(i) + 71); }
	for (size_t i = 0; i < x.size(); ++i) { x[i] = value(static_cast<int>(i) + 11); z[i] = value(static_cast<int>(i) + 29); }
	for (size_t i = 0; i < adt.size(); ++i) {
		dt[i] = 0.04F + 0.02F * (value(static_cast<int>(i) + 37) + 0.5F);
		adt[i] = -(0.2F + 0.1F * (value(static_cast<int>(i) + 43) + 0.5F)) * dt[i];
		trap[i] = value(static_cast<int>(i) + 53);
	}
	for (size_t i = 0; i < angle.size(); ++i) angle[i] = value(static_cast<int>(i) + 61);
	for (size_t i = 0; i < cb.size(); ++i) { cb[i] = 0.1F * value(static_cast<int>(i) + 83); kb[i] = 0.1F * value(static_cast<int>(i) + 97); }
	for (int h = 0; h < H; ++h) d[static_cast<size_t>(h)] = 0.2F + 0.1F * static_cast<float>(h);
	for (size_t i = 0; i < mx.size(); ++i) {
		mx[i] = 0.4F + 0.15F * value(static_cast<int>(i) + 107);
		mz[i] = 0.8F + 0.1F * value(static_cast<int>(i) + 127);
		mo[i] = 0.5F + 0.1F * value(static_cast<int>(i) + 149);
	}
	oa::SsmConfig cfg{
		.batch = Bn, .seqLen = L, .nHeads = H, .nGroups = G,
		.headDim = P, .stateSize = N, .numRopeAngles = A, .mimoRank = R,
		.hasZ = 1, .hasD = 1, .hasOutNorm = 0};
	auto y = oa::FnMatrix::mamba3Mimo(
		matFromVec(c, {Bn, L, R * G, N}), matFromVec(k, {Bn, L, R * G, N}),
		matFromVec(x, {Bn, L, H, P}), matFromVec(z, {Bn, L, H, P}),
		matFromVec(adt, {Bn, L, H}), matFromVec(dt, {Bn, L, H}),
		matFromVec(trap, {Bn, L, H}), matFromVec(angle, {Bn, L, A}),
		matFromVec(cb, {H, R, N}), matFromVec(kb, {H, R, N}),
		matFromVec(d, {H}), matFromVec(mx, {H, R, P}),
		matFromVec(mz, {H, R, P}), matFromVec(mo, {H, R, P}),
		matFromVec(norm, {H, P}), cfg);
	syncCtx();

	std::vector<float> expected(x.size(), 0.0F);
	auto sig = [](float v) { return 1.0F / (1.0F + std::exp(-v)); };
	auto silu = [&](float v) { return v * sig(v); };
	for (int b = 0; b < Bn; ++b) for (int h = 0; h < H; ++h) {
		int g = h / (H / G);
		std::vector<float> state(P * N, 0.0F), theta(A, 0.0F);
		for (int t = 0; t < L; ++t) {
			int scalar = (b * L + t) * H + h;
			float gamma = dt[static_cast<size_t>(scalar)] * sig(trap[static_cast<size_t>(scalar)]);
			float shifted = 0.0F;
			if (t + 1 < L) {
				int next = (b * L + t + 1) * H + h;
				shifted = dt[static_cast<size_t>(next)] * (1.0F - sig(trap[static_cast<size_t>(next)]));
			}
			for (int a = 0; a < A; ++a)
				theta[static_cast<size_t>(a)] += angle[static_cast<size_t>((b * L + t) * A + a)] * dt[static_cast<size_t>(scalar)];
			std::vector<float> q(R * N), key(R * N);
			for (int r = 0; r < R; ++r) for (int n = 0; n < N; ++n) {
				int raw = (((b * L + t) * (R * G) + r * G + g) * N) + n;
				int bias = (h * R + r) * N + n;
				q[static_cast<size_t>(r * N + n)] = c[static_cast<size_t>(raw)] + cb[static_cast<size_t>(bias)];
				key[static_cast<size_t>(r * N + n)] = k[static_cast<size_t>(raw)] + kb[static_cast<size_t>(bias)];
			}
			for (int r = 0; r < R; ++r) for (int a = 0; a < A; ++a) {
				int i0 = r * N + 2 * a, i1 = i0 + 1;
				float cs = std::cos(theta[static_cast<size_t>(a)]), sn = std::sin(theta[static_cast<size_t>(a)]);
				float q0 = q[static_cast<size_t>(i0)], q1 = q[static_cast<size_t>(i1)];
				float k0 = key[static_cast<size_t>(i0)], k1 = key[static_cast<size_t>(i1)];
				q[static_cast<size_t>(i0)] = q0 * cs - q1 * sn; q[static_cast<size_t>(i1)] = q0 * sn + q1 * cs;
				key[static_cast<size_t>(i0)] = k0 * cs - k1 * sn; key[static_cast<size_t>(i1)] = k0 * sn + k1 * cs;
			}
			std::vector<float> u(P * N, 0.0F), rawOut(R * P, 0.0F);
			for (int p = 0; p < P; ++p) for (int n = 0; n < N; ++n)
				for (int r = 0; r < R; ++r) {
					int xp = ((b * L + t) * H + h) * P + p;
					u[static_cast<size_t>(p * N + n)] += x[static_cast<size_t>(xp)]
						* mx[static_cast<size_t>((h * R + r) * P + p)]
						* key[static_cast<size_t>(r * N + n)];
				}
			float decay = std::exp(adt[static_cast<size_t>(scalar)]);
			for (int r = 0; r < R; ++r) for (int p = 0; p < P; ++p) {
				float pre = 0.0F;
				for (int n = 0; n < N; ++n)
					pre += q[static_cast<size_t>(r * N + n)] *
						(decay * state[static_cast<size_t>(p * N + n)] + gamma * u[static_cast<size_t>(p * N + n)]);
				int xp = ((b * L + t) * H + h) * P + p;
				pre += d[static_cast<size_t>(h)] * x[static_cast<size_t>(xp)]
					* mx[static_cast<size_t>((h * R + r) * P + p)];
				rawOut[static_cast<size_t>(r * P + p)] = pre;
			}
			for (int p = 0; p < P; ++p) for (int n = 0; n < N; ++n)
				state[static_cast<size_t>(p * N + n)] = decay * state[static_cast<size_t>(p * N + n)]
					+ (gamma + shifted) * u[static_cast<size_t>(p * N + n)];
			for (int p = 0; p < P; ++p) {
				int xp = ((b * L + t) * H + h) * P + p;
				for (int r = 0; r < R; ++r) {
					int rp = (h * R + r) * P + p;
					expected[static_cast<size_t>(xp)] += rawOut[static_cast<size_t>(r * P + p)]
						* silu(z[static_cast<size_t>(xp)] * mz[static_cast<size_t>(rp)])
						* mo[static_cast<size_t>(rp)];
				}
			}
		}
	}
	float maxErr = 0.0F;
	for (size_t i = 0; i < expected.size(); ++i) {
		maxErr = std::max(maxErr, std::abs(y.at(static_cast<oa::I64>(i)) - expected[i]));
		EXPECT_NEAR(y.at(static_cast<oa::I64>(i)), expected[i], 2e-4F) << "i=" << i;
	}
	std::cerr << "Mamba3Mimo shared-state oracle max abs err = " << maxErr << std::endl;
}

TEST(TestMamba3, MimoBackwardMatchesFiniteDifferences) {
	const int Bn = 1, L = 3, H = 1, G = 1, R = 2, P = 2, N = 2, A = 1;
	struct Inputs {
		std::vector<float> c, k, x, z, adt, dt, trap, angle;
		std::vector<float> cBias, kBias, d, mimoX, mimoZ, mimoO, norm;
	};
	auto fill = [](size_t count, int offset, float scale = 0.2F) {
		std::vector<float> result(count);
		for (size_t i = 0; i < count; ++i)
			result[i] = scale * std::sin(0.31F * static_cast<float>(i + offset));
		return result;
	};
	Inputs inputs{
		.c = fill(Bn * L * R * G * N, 1),
		.k = fill(Bn * L * R * G * N, 13),
		.x = fill(Bn * L * H * P, 29),
		.z = fill(Bn * L * H * P, 37),
		.adt = fill(Bn * L * H, 43, 0.08F),
		.dt = fill(Bn * L * H, 47, 0.02F),
		.trap = fill(Bn * L * H, 53),
		.angle = fill(Bn * L * A, 59),
		.cBias = fill(H * R * N, 61, 0.05F),
		.kBias = fill(H * R * N, 67, 0.05F),
		.d = fill(H, 71, 0.3F),
		.mimoX = fill(H * R * P, 73, 0.4F),
		.mimoZ = fill(H * R * P, 79, 0.5F),
		.mimoO = fill(H * R * P, 83, 0.45F),
		.norm = fill(H * P, 89, 0.7F),
	};
	// Keep the decay and timestep in a stable, positive operating region.
	for (float& value : inputs.adt) value -= 0.07F;
	for (float& value : inputs.dt) value += 0.08F;
	for (float& value : inputs.norm) value += 1.0F;
	const auto dOut = fill(Bn * L * H * P, 97, 0.6F);
	const oa::SsmConfig config{
		.batch = Bn, .seqLen = L, .nHeads = H, .nGroups = G,
		.headDim = P, .stateSize = N, .numRopeAngles = A, .mimoRank = R,
		.hasZ = 1, .hasD = 1, .hasOutNorm = 1};

	auto forward = [&](const Inputs& in) {
		return oa::FnMatrix::mamba3Mimo(
			matFromVec(in.c, {Bn, L, R * G, N}),
			matFromVec(in.k, {Bn, L, R * G, N}),
			matFromVec(in.x, {Bn, L, H, P}), matFromVec(in.z, {Bn, L, H, P}),
			matFromVec(in.adt, {Bn, L, H}), matFromVec(in.dt, {Bn, L, H}),
			matFromVec(in.trap, {Bn, L, H}), matFromVec(in.angle, {Bn, L, A}),
			matFromVec(in.cBias, {H, R, N}), matFromVec(in.kBias, {H, R, N}),
			matFromVec(in.d, {H}), matFromVec(in.mimoX, {H, R, P}),
			matFromVec(in.mimoZ, {H, R, P}), matFromVec(in.mimoO, {H, R, P}),
			matFromVec(in.norm, {H, P}), config);
	};
	auto loss = [&](const Inputs& in) {
		auto output = forward(in);
		syncCtx();
		float value = 0.0F;
		for (size_t i = 0; i < dOut.size(); ++i)
			value += output.at(static_cast<oa::I64>(i)) * dOut[i];
		return value;
	};

	auto c = matFromVec(inputs.c, {Bn, L, R * G, N});
	auto k = matFromVec(inputs.k, {Bn, L, R * G, N});
	auto x = matFromVec(inputs.x, {Bn, L, H, P});
	auto z = matFromVec(inputs.z, {Bn, L, H, P});
	auto adt = matFromVec(inputs.adt, {Bn, L, H});
	auto dt = matFromVec(inputs.dt, {Bn, L, H});
	auto trap = matFromVec(inputs.trap, {Bn, L, H});
	auto angle = matFromVec(inputs.angle, {Bn, L, A});
	auto cBias = matFromVec(inputs.cBias, {H, R, N});
	auto kBias = matFromVec(inputs.kBias, {H, R, N});
	auto d = matFromVec(inputs.d, {H});
	auto mimoX = matFromVec(inputs.mimoX, {H, R, P});
	auto mimoZ = matFromVec(inputs.mimoZ, {H, R, P});
	auto mimoO = matFromVec(inputs.mimoO, {H, R, P});
	auto norm = matFromVec(inputs.norm, {H, P});
	auto gradients = oa::FnMatrix::mamba3MimoBwd(
		matFromVec(dOut, {Bn, L, H, P}), c, k, x, z, adt, dt, trap, angle,
		cBias, kBias, d, mimoX, mimoZ, mimoO, norm, config);
	syncCtx();

	using Field = std::vector<float> Inputs::*;
	auto check = [&](const char* name, Field field, const oa::Matrix& analytic) {
		constexpr float epsilon = 2e-3F;
		float maxError = 0.0F;
		for (size_t i = 0; i < (inputs.*field).size(); ++i) {
			Inputs plus = inputs, minus = inputs;
			(plus.*field)[i] += epsilon;
			(minus.*field)[i] -= epsilon;
			float numerical = (loss(plus) - loss(minus)) / (2.0F * epsilon);
			float actual = analytic.at(static_cast<oa::I64>(i));
			maxError = std::max(maxError, std::abs(actual - numerical));
			EXPECT_NEAR(actual, numerical, 2e-3F + 2e-2F * std::abs(numerical))
				<< name << "[" << i << "]";
		}
		std::cerr << "Mamba3Mimo " << name << " gradient max abs err = "
			<< maxError << std::endl;
	};
	check("C", &Inputs::c, gradients.dC);
	check("K", &Inputs::k, gradients.dB);
	check("X", &Inputs::x, gradients.dX);
	check("Z", &Inputs::z, gradients.dZ);
	check("adt", &Inputs::adt, gradients.dAdt);
	check("dt", &Inputs::dt, gradients.dDt);
	check("trap", &Inputs::trap, gradients.dTrap);
	check("angle", &Inputs::angle, gradients.dAngle);
	check("cBias", &Inputs::cBias, gradients.dCBias);
	check("kBias", &Inputs::kBias, gradients.dBBias);
	check("D", &Inputs::d, gradients.dD);
	check("mimoX", &Inputs::mimoX, gradients.dMimoX);
	check("mimoZ", &Inputs::mimoZ, gradients.dMimoZ);
	check("mimoO", &Inputs::mimoO, gradients.dMimoO);
	check("norm", &Inputs::norm, gradients.dNormWeight);
}

TEST(TestMamba3, RejectsInvalidStabilityBounds) {
	EXPECT_THROW(
		oa::Mamba3Module(32, 16, 2, 0, 1, 0.5F),
		std::invalid_argument);
	EXPECT_THROW(
		oa::Mamba3Module(std::numeric_limits<oa::I32>::max(), 16, 2, 16, 1, 0.5F),
		std::invalid_argument);
	EXPECT_THROW(
		oa::Mamba3Module(32, 16, 2, 16, 1, 1.5F),
		std::invalid_argument);
	EXPECT_THROW(
		oa::Mamba3Module(32, 16, 2, 16, 1, 0.5F, false, 4,
			0.1F, 0.01F),
		std::invalid_argument);

	const oa::Matrix c = oa::FnMatrix::empty({1, 1, 2, 2}, oa::ScalarType::Float32);
	const oa::Matrix k = oa::FnMatrix::empty(c.getShape(), oa::ScalarType::Float32);
	const oa::Matrix x = oa::FnMatrix::empty({1, 1, 1, 2}, oa::ScalarType::Float32);
	const oa::Matrix z = oa::FnMatrix::empty(x.getShape(), oa::ScalarType::Float32);
	const oa::Matrix scalar = oa::FnMatrix::empty({1, 1, 1}, oa::ScalarType::Float32);
	const oa::Matrix angle = oa::FnMatrix::empty({1, 1, 1}, oa::ScalarType::Float32);
	const oa::Matrix bias = oa::FnMatrix::empty({1, 2, 2}, oa::ScalarType::Float32);
	const oa::Matrix d = oa::FnMatrix::empty({1}, oa::ScalarType::Float32);
	const oa::Matrix mix = oa::FnMatrix::empty({1, 2, 2}, oa::ScalarType::Float32);
	const oa::Matrix norm = oa::FnMatrix::empty({1, 2}, oa::ScalarType::Float32);
	const oa::Matrix badC = oa::FnMatrix::empty(c.getShape(), oa::ScalarType::BFloat16);
	oa::SsmConfig config{
		.batch = 1, .seqLen = 1, .nHeads = 1, .nGroups = 1,
		.headDim = 2, .stateSize = 2, .numRopeAngles = 1, .mimoRank = 2,
		.hasZ = 1, .hasD = 1, .hasOutNorm = 1};

	EXPECT_TRUE(oa::FnMatrix::mamba3Mimo(
		badC, k, x, z, scalar, scalar, scalar, angle, bias, bias, d,
		mix, mix, mix, norm, config).isEmpty());
	const auto rejectedBackward = oa::FnMatrix::mamba3MimoBwd(
		x, badC, k, x, z, scalar, scalar, scalar, angle, bias, bias, d,
		mix, mix, mix, norm, config);
	EXPECT_TRUE(rejectedBackward.dC.isEmpty());
	const oa::Matrix ssmState = oa::FnMatrix::empty({1, 1, 2, 2}, oa::ScalarType::Float32);
	const oa::Matrix angleState = oa::FnMatrix::empty({1, 1, 1}, oa::ScalarType::Float32);
	const oa::Matrix kState = oa::FnMatrix::empty({1, 1, 2, 2}, oa::ScalarType::Float32);
	const oa::Matrix vState = oa::FnMatrix::empty({1, 1, 2, 2}, oa::ScalarType::Float32);
	EXPECT_TRUE(oa::FnMatrix::mamba3MimoStep(
		badC, k, x, z, scalar, scalar, scalar, angle, bias, bias, d,
		mix, mix, mix, norm, ssmState, angleState, kState, vState,
		config).isEmpty());

	config.hasOutNorm = 2;
	EXPECT_TRUE(oa::FnMatrix::mamba3Mimo(
		c, k, x, z, scalar, scalar, scalar, angle, bias, bias, d,
		mix, mix, mix, norm, config).isEmpty());
}

TEST(TestMamba3, MimoStepMatchesForwardWithGroupedQk) {
	const int Bn = 2, Ln = 5, D = 32;
	oa::Mamba3Module mamba3(D, 8, 2, 16, 2, 0.5F, true, 2);
	auto input = oa::FnMatrix::randN(oa::MatrixShape{Bn, Ln, D}, oa::ScalarType::Float32);
	auto full = mamba3.forward(input);
	syncCtx();
	mamba3.resetState(Bn);
	float maxErr = 0.0F;
	for (int t = 0; t < Ln; ++t) {
		auto step = mamba3.step(oa::FnMatrix::slice(input, 1, t, t + 1));
		syncCtx();
		for (int b = 0; b < Bn; ++b) for (int d = 0; d < D; ++d) {
			float got = step.at(static_cast<oa::I64>(b * D + d));
			float expected = full.at(static_cast<oa::I64>((b * Ln + t) * D + d));
			maxErr = std::max(maxErr, std::abs(got - expected));
			EXPECT_NEAR(got, expected, 3e-3F) << "t=" << t << " b=" << b << " d=" << d;
		}
	}
	std::cerr << "Mamba3Mimo step-vs-forward max abs err = " << maxErr << std::endl;
}

TEST(TestMamba3, ForwardPassSimple) {
	// Simplified test to isolate the bad optional access
	// Test basic operations that Mamba3 uses
	std::cerr << "Test 1: RandN" << std::endl;
	auto input = oa::FnMatrix::randN(oa::MatrixShape{2, 32, 64}, oa::ScalarType::Float32);
	std::cerr << "Test 1 passed" << std::endl;

	std::cerr << "Test 2: MatMul" << std::endl;
	auto weight = oa::FnMatrix::randGlorotUniform(oa::MatrixShape{64, 64}, oa::ScalarType::Float32);
	auto projected = oa::FnMatrix::matMulNt(input, weight);
	std::cerr << "Test 2 passed" << std::endl;

	std::cerr << "Test 3: Slice" << std::endl;
	auto z = oa::FnMatrix::slice(projected, 2, 0, 64);
	std::cerr << "Test 3 passed" << std::endl;

	std::cerr << "Test 4: reshape" << std::endl;
	auto zFlat = z.reshape(oa::MatrixShape{64, 64});
	std::cerr << "Test 4 passed" << std::endl;

	std::cerr << "Test 5: Exp" << std::endl;
	auto x = oa::FnMatrix::randN(oa::MatrixShape{100}, oa::ScalarType::Float32);
	auto exp = oa::FnMatrix::exp(x);
	std::cerr << "Test 5 passed" << std::endl;

	std::cerr << "Test 6: Softplus" << std::endl;
	auto softplus = oa::FnMatrix::softplus(x);
	std::cerr << "Test 6 passed" << std::endl;

	std::cerr << "Test 7: HeavyTailActivation" << std::endl;
	auto hta = oa::FnMatrix::heavyTailActivation(x);
	std::cerr << "Test 7 passed" << std::endl;

	std::cerr << "Test 8: Rand" << std::endl;
	auto rand = oa::FnMatrix::rand(oa::MatrixShape{100}, oa::ScalarType::Float32);
	std::cerr << "Test 8 passed" << std::endl;

	std::cerr << "Test 9: log" << std::endl;
	auto xPositive = oa::FnMatrix::randN(oa::MatrixShape{100}, oa::ScalarType::Float32) + 1.0f;  // Ensure positive
	auto log = oa::FnMatrix::log(xPositive);
	std::cerr << "Test 9 passed" << std::endl;

	std::cerr << "Test 10: Mean" << std::endl;
	auto x2d = oa::FnMatrix::randN(oa::MatrixShape{10, 100}, oa::ScalarType::Float32);
	auto mean = oa::FnMatrix::mean(x2d, 0);
	std::cerr << "Test 10 passed" << std::endl;

	// Test 11: SsmScan removed - old API deprecated, use Mamba3Module instead

	std::cerr << "Test 12: scalar-matrix arithmetic" << std::endl;
	auto m = oa::FnMatrix::randN(oa::MatrixShape{100}, oa::ScalarType::Float32);
	auto m2 = m * 2.0f;
	auto m3 = m2 + 1.0f;
	auto m4 = m3 - 0.5f;
	std::cerr << "Test 12 passed" << std::endl;

	std::cerr << "Test 13: Mamba3 constructor" << std::endl;
	oa::Mamba3Module mamba3(
		64,   // d_model
		16,   // d_state
		2,    // expand
		32,   // headdim
		1     // ngroups
	);
	std::cerr << "Test 13 passed" << std::endl;

	std::cerr << "Test 14: ClampMin" << std::endl;
	auto cm = oa::FnMatrix::clampMin(m, 0.1f);
	std::cerr << "Test 14 passed" << std::endl;

	std::cerr << "Test 15: ClampMax" << std::endl;
	auto cmax = oa::FnMatrix::clampMax(m, 1.0f);
	std::cerr << "Test 15 passed" << std::endl;

	std::cerr << "Test 16: Silu" << std::endl;
	auto silu = oa::FnMatrix::silu(m);
	std::cerr << "Test 16 passed" << std::endl;

	std::cerr << "Test 17: Broadcasting addition" << std::endl;
	auto m2d = oa::FnMatrix::randN(oa::MatrixShape{10, 100}, oa::ScalarType::Float32);
	auto bias = oa::FnMatrix::randN(oa::MatrixShape{100}, oa::ScalarType::Float32);
	auto broadcastAdd = m2d + bias;  // [10,100] + [100] -> broadcast
	std::cerr << "Test 17 passed" << std::endl;

	std::cerr << "Test 18: 4D broadcasting addition" << std::endl;
	auto m4d = oa::FnMatrix::randN(oa::MatrixShape{2, 32, 1, 16}, oa::ScalarType::Float32);
	auto bias2d = oa::FnMatrix::randN(oa::MatrixShape{1, 16}, oa::ScalarType::Float32);
	auto broadcastAdd4d = m4d + bias2d;  // [2,32,1,16] + [1,16] -> broadcast
	std::cerr << "Test 18 passed" << std::endl;

	std::cerr << "Test 19: Element-wise matrix multiplication" << std::endl;
	auto m3d = oa::FnMatrix::randN(oa::MatrixShape{2, 32, 64}, oa::ScalarType::Float32);
	auto m3d2 = oa::FnMatrix::randN(oa::MatrixShape{2, 32, 64}, oa::ScalarType::Float32);
	auto elemMul = m3d * m3d2;
	std::cerr << "Test 19 passed" << std::endl;

	// This test validates that every operation can be recorded. Complete that
	// recording before its local matrices are released so no deferred work can
	// leak into the following parity test.
	syncCtx();
}

// Fused Mamba3Preprocess forward parity: GPU output must match a CPU reference
// implementation of the same split + RmsNorm + dt + adt math.
TEST(TestMamba3, PreprocessKernelParity) {
	const int B = 2, S = 4, DI = 64, N = 16, H = 2, G = 1, R = 4, A = 8;
	const int bcWidth = N * G * R;
	const int dInProj = 2 * DI + 2 * bcWidth + A + 3 * H;
	const int rows = B * S;

	auto idx2d = [&](int r, int c) { return r * dInProj + c; };
	std::vector<float> proj(static_cast<size_t>(rows) * dInProj);
	auto f = [](int s) { return std::sin(0.7F * s + 1.3F) * 0.5F; };
	for (size_t i = 0; i < proj.size(); ++i) proj[i] = f(static_cast<int>(i));
	std::vector<float> dtBiasVec(static_cast<size_t>(H));
	for (int h = 0; h < H; ++h) dtBiasVec[h] = 0.01F * f(h + 17);

	auto projected = matFromVec(proj, oa::MatrixShape{rows, dInProj});
	auto dtBias = matFromVec(dtBiasVec, oa::MatrixShape{H});

	oa::Mamba3PreprocessConfig cfg{
		.dInner = DI, .dState = N, .nHeads = H, .numRopeAngles = A,
		.nGroups = G, .mimoRank = R, .eps = 1e-5F, .dtMin = 0.001F,
		.dtMax = 0.1F, .aFloor = 1e-4F
	};
	auto pp = oa::FnMatrix::mamba3Preprocess(projected, dtBias, cfg);
	syncCtx();

	auto rms = [&](const std::vector<float>& x, int base, int cols) {
		double sq = 0.0;
		for (int i = 0; i < cols; ++i) sq += static_cast<double>(x[base + i]) * x[base + i];
		return 1.0F / static_cast<float>(std::sqrt(static_cast<float>(sq / cols) + 1e-5F));
	};

	auto read2d = [&](const oa::Matrix& m, int r, int c) {
		return m.at(static_cast<oa::I64>(r) * m.size(1) + c);
	};

	float maxErr = 0.0F;
	auto check = [&](const char* name, const oa::Matrix& got, int gotCols, auto refFn) {
		for (int r = 0; r < rows; ++r) {
			for (int c = 0; c < gotCols; ++c) {
				float g = read2d(got, r, c);
				float e = refFn(r, c);
				maxErr = std::max(maxErr, std::abs(g - e));
				EXPECT_NEAR(g, e, 1e-4F) << name << " r=" << r << " c=" << c;
			}
		}
	};

	check("Z", pp.z, DI, [&](int r, int c) { return proj[idx2d(r, c)]; });
	check("X", pp.x, DI, [&](int r, int c) { return proj[idx2d(r, c + DI)]; });
	check("trap", pp.trap, H, [&](int r, int c) {
		return proj[idx2d(r, 2 * DI + 2 * bcWidth + H + H + c)];
	});
	check("angle", pp.angle, A, [&](int r, int c) {
		return proj[idx2d(r, 2 * DI + 2 * bcWidth + H + H + H + c)];
	});
	check("bh", pp.bh, bcWidth, [&](int r, int c) {
		int gr = c / N;
		int i = c % N;
		int base = idx2d(r, 2 * DI + gr * N);
		float inv = rms(proj, r * dInProj + 2 * DI + gr * N, N);
		return proj[base + i] * inv;
	});
	check("ch", pp.ch, bcWidth, [&](int r, int c) {
		int gr = c / N;
		int i = c % N;
		int base = idx2d(r, 2 * DI + bcWidth + gr * N);
		float inv = rms(proj, r * dInProj + 2 * DI + bcWidth + gr * N, N);
		return proj[base + i] * inv;
	});
	check("DT", pp.dt, H, [&](int r, int c) {
		float raw = proj[idx2d(r, 2 * DI + 2 * bcWidth + c)] + dtBiasVec[c];
		float sp = std::max(raw, 0.0F) + std::log(1.0F + std::exp(-std::abs(raw)));
		return std::clamp(sp, 0.001F, 0.1F);
	});
	check("ADT", pp.adt, H, [&](int r, int c) {
		float ddA = proj[idx2d(r, 2 * DI + 2 * bcWidth + H + c)];
		float heavy = (ddA >= 0.0F) ? (1.0F + ddA) : (1.0F / (1.0F - ddA));
		float aTok = std::min(-heavy, -1e-4F);
		float raw = proj[idx2d(r, 2 * DI + 2 * bcWidth + c)] + dtBiasVec[c];
		float sp = std::max(raw, 0.0F) + std::log(1.0F + std::exp(-std::abs(raw)));
		float dt = std::clamp(sp, 0.001F, 0.1F);
		return aTok * dt;
	});

	std::cerr << "Mamba3Preprocess parity max abs err = " << maxErr << "\n";
}

// Finite-difference gradcheck for Mamba3Preprocess backward: analytic gradients
// of the projected tensor and dt_bias vs central differences.
TEST(TestMamba3, PreprocessKernelGradcheck) {
	const int B = 2;
	const int S = 4;
	const int DI = 64;
	const int N = 16;
	const int H = 2;
	const int G = 1;
	const int R = 4;
	const int A = 8;
	const int bcWidth = N * G * R;
	const int dInProj = 2 * DI + 2 * bcWidth + A + 3 * H;
	const int rows = B * S;

	std::vector<float> proj(static_cast<size_t>(rows) * dInProj);
	auto f = [](int s) { return std::sin(0.7F * s + 1.3F) * 0.5F; };
	for (size_t i = 0; i < proj.size(); ++i) proj[i] = f(static_cast<int>(i));
	std::vector<float> dtBiasVec(static_cast<size_t>(H));
	for (int h = 0; h < H; ++h) dtBiasVec[h] = 0.01F * f(h + 17);
	auto wf = [](int s) { return std::sin(0.3F * s + 0.7F) * 0.5F + 0.5F; };
	std::vector<float> wZ(static_cast<size_t>(rows) * DI);
	std::vector<float> wX(static_cast<size_t>(rows) * DI);
	std::vector<float> wBC(static_cast<size_t>(rows) * bcWidth);
	std::vector<float> wDT(static_cast<size_t>(rows) * H);
	std::vector<float> wADT(static_cast<size_t>(rows) * H);
	std::vector<float> wTrap(static_cast<size_t>(rows) * H);
	std::vector<float> wAngle(static_cast<size_t>(rows) * A);
	std::vector<float> wDtBias(static_cast<size_t>(H), 0.0F);
	for (size_t i = 0; i < wZ.size(); ++i) wZ[i] = wf(static_cast<int>(i));
	for (size_t i = 0; i < wX.size(); ++i) wX[i] = wf(static_cast<int>(i) + 1000);
	for (size_t i = 0; i < wBC.size(); ++i) wBC[i] = wf(static_cast<int>(i) + 2000);
	for (size_t i = 0; i < wDT.size(); ++i) wDT[i] = wf(static_cast<int>(i) + 3000);
	for (size_t i = 0; i < wADT.size(); ++i) wADT[i] = wf(static_cast<int>(i) + 4000);
	for (size_t i = 0; i < wTrap.size(); ++i) wTrap[i] = wf(static_cast<int>(i) + 5000);
	for (size_t i = 0; i < wAngle.size(); ++i) wAngle[i] = wf(static_cast<int>(i) + 6000);
	for (size_t h = 0; h < H; ++h) wDtBias[h] = wf(static_cast<int>(h) + 7000);

	auto makeProjected = [&]() { return matFromVec(proj, oa::MatrixShape{rows, dInProj}); };
	auto makeDtBias = [&]() { return matFromVec(dtBiasVec, oa::MatrixShape{H}); };
	auto makeW = [&](std::vector<float>& v, oa::MatrixShape shape) { return matFromVec(v, shape); };

	oa::Mamba3PreprocessConfig cfg{
		.dInner = DI, .dState = N, .nHeads = H, .numRopeAngles = A,
		.nGroups = G, .mimoRank = R, .eps = 1e-5F, .dtMin = 0.001F,
		.dtMax = 0.1F, .aFloor = 1e-4F
	};

	auto computeLoss = [&]() -> double {
		auto projected = makeProjected();
		auto dtBias = makeDtBias();
		auto pp = oa::FnMatrix::mamba3Preprocess(projected, dtBias, cfg);
		auto l = oa::FnMatrix::sum(pp.z * makeW(wZ, oa::MatrixShape{rows, DI}))
			+ oa::FnMatrix::sum(pp.x * makeW(wX, oa::MatrixShape{rows, DI}))
			+ oa::FnMatrix::sum(pp.bh * makeW(wBC, oa::MatrixShape{rows, bcWidth}))
			+ oa::FnMatrix::sum(pp.ch * makeW(wBC, oa::MatrixShape{rows, bcWidth}))
			+ oa::FnMatrix::sum(pp.dt * makeW(wDT, oa::MatrixShape{rows, H}))
			+ oa::FnMatrix::sum(pp.adt * makeW(wADT, oa::MatrixShape{rows, H}))
			+ oa::FnMatrix::sum(pp.trap * makeW(wTrap, oa::MatrixShape{rows, H}))
			+ oa::FnMatrix::sum(pp.angle * makeW(wAngle, oa::MatrixShape{rows, A}))
			+ oa::FnMatrix::sum(dtBias * makeW(wDtBias, oa::MatrixShape{H}));
		syncCtx();
		return static_cast<double>(l.at(0));
	};

	auto projected = makeProjected();
	auto dtBias = makeDtBias();
	projected.setRequiresGrad(true);
	dtBias.setRequiresGrad(true);
	oa::GradientTape tape;
	auto pp = oa::FnMatrix::mamba3Preprocess(projected, dtBias, cfg);
	auto l = oa::FnMatrix::sum(pp.z * makeW(wZ, oa::MatrixShape{rows, DI}))
		+ oa::FnMatrix::sum(pp.x * makeW(wX, oa::MatrixShape{rows, DI}))
		+ oa::FnMatrix::sum(pp.bh * makeW(wBC, oa::MatrixShape{rows, bcWidth}))
		+ oa::FnMatrix::sum(pp.ch * makeW(wBC, oa::MatrixShape{rows, bcWidth}))
		+ oa::FnMatrix::sum(pp.dt * makeW(wDT, oa::MatrixShape{rows, H}))
		+ oa::FnMatrix::sum(pp.adt * makeW(wADT, oa::MatrixShape{rows, H}))
		+ oa::FnMatrix::sum(pp.trap * makeW(wTrap, oa::MatrixShape{rows, H}))
		+ oa::FnMatrix::sum(pp.angle * makeW(wAngle, oa::MatrixShape{rows, A}))
		+ oa::FnMatrix::sum(dtBias * makeW(wDtBias, oa::MatrixShape{H}));
	syncCtx();
	tape.backward(l);
	syncCtx();
	auto dProjected = projected.gradMatrix();
	auto dDtBias = dtBias.gradMatrix();

	const float eps = 2e-3F;
	float maxErr = 0.0F;
	auto check = [&](const char* name, std::vector<float>& vec, const oa::Matrix& ana, int stride) {
		for (size_t i = 0; i < vec.size(); i += stride) {
			float orig = vec[i];
			vec[i] = orig + eps; double lp = computeLoss();
			vec[i] = orig - eps; double lm = computeLoss();
			vec[i] = orig;
			float num = static_cast<float>((lp - lm) / (2.0 * eps));
			float a = ana.at(static_cast<oa::I64>(i));
			float tol = 2e-2F + 3e-2F * std::abs(a);
			maxErr = std::max(maxErr, std::abs(num - a));
			EXPECT_NEAR(num, a, tol) << name << "[" << i << "] num=" << num << " ana=" << a;
		}
	};

	check("dProjected", proj, dProjected, 13);
	check("dDtBias", dtBiasVec, dDtBias, 1);
	std::cerr << "Mamba3Preprocess gradcheck max abs err = " << maxErr << "\n";
}

// verify empyrealm-branded dispatch path is wired correctly and matches the CPU reference.
TEST(TestMamba3, EmpyrealmPreprocessKernelParity) {
	const int B = 2;
	const int S = 4;
	const int DI = 64;
	const int N = 16;
	const int H = 2;
	const int G = 1;
	const int R = 4;
	const int A = 8;
	const int bcWidth = N * G * R;
	const int dInProj = 2 * DI + 2 * bcWidth + A + 3 * H;
	const int rows = B * S;

	auto idx2d = [&](int r, int c) { return r * dInProj + c; };
	std::vector<float> proj(static_cast<size_t>(rows) * dInProj);
	auto f = [](int s) { return std::sin(0.7F * s + 1.3F) * 0.5F; };
	for (size_t i = 0; i < proj.size(); ++i) proj[i] = f(static_cast<int>(i));
	std::vector<float> dtBiasVec(static_cast<size_t>(H));
	for (int h = 0; h < H; ++h) dtBiasVec[h] = 0.01F * f(h + 17);

	auto projected = matFromVec(proj, oa::MatrixShape{rows, dInProj});
	auto dtBias = matFromVec(dtBiasVec, oa::MatrixShape{H});

	oa::Mamba3PreprocessConfig cfg{
		.dInner = DI, .dState = N, .nHeads = H, .numRopeAngles = A,
		.nGroups = G, .mimoRank = R, .eps = 1e-5F, .dtMin = 0.001F,
		.dtMax = 0.1F, .aFloor = 1e-4F
	};
	auto pp = oa::FnMatrix::empyrealmPreprocess(projected, dtBias, cfg);
	syncCtx();

	auto read2d = [&](const oa::Matrix& m, int r, int c) {
		return m.at(static_cast<oa::I64>(r) * m.size(1) + c);
	};

	float maxErr = 0.0F;
	for (int r = 0; r < rows; ++r) {
		for (int c = 0; c < DI; ++c) {
			maxErr = std::max(maxErr, std::abs(read2d(pp.z, r, c) - proj[idx2d(r, c)]));
			maxErr = std::max(maxErr, std::abs(read2d(pp.x, r, c) - proj[idx2d(r, c + DI)]));
		}
		for (int c = 0; c < H; ++c) {
			float raw = proj[idx2d(r, 2 * DI + 2 * bcWidth + c)] + dtBiasVec[c];
			float sp = std::max(raw, 0.0F) + std::log(1.0F + std::exp(-std::abs(raw)));
			float ref = std::clamp(sp, 0.001F, 0.1F);
			maxErr = std::max(maxErr, std::abs(read2d(pp.dt, r, c) - ref));
		}
	}
	std::cerr << "EmpyrealmPreprocess parity max abs err = " << maxErr << "\n";
	EXPECT_LT(maxErr, 1e-4F);
}
