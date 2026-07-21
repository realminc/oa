// OaAutogradFnMatrix — Gradient nodes for matrix / activation / conv / norm / pool ops.
//
// Simple element-wise and activation grads are auto-generated from schema TOML:
//   - AutogradElemwise.gen.h  (Core/FnMatrix elemwise ops)
//   - AutogradActivation.gen.h (Ml/FnMatrix activation ops)
//
// Complex grads (Linear, Conv, Norm, Pool, etc.) remain hand-written below.

#pragma once

#include <Oa/Core/FnMatrix.h>
#include <Oa/Ml/Autograd.h>
#include <Oa/Ml/FnMatrix.h>
#include <Oa/Runtime/Context.h>

// ─── Element-wise manual (complex or no schema entry) ───────────────────────

class OaGradDiv final : public OaGradNode {
public:
	void Backward(const OaMatrix& InDOut, OaVec<OaMatrix>& OutDIn) override {
		const OaMatrix& a = Saved_[0];
		const OaMatrix& b = Saved_[1];
		if (OutDIn.Size() > 0) OutDIn[0] = OaFnMatrix::Div(InDOut, b);
		if (OutDIn.Size() > 1) {
			auto aDivB = OaFnMatrix::Div(a, b);
			auto dOutDivB = OaFnMatrix::Div(InDOut, b);
			OutDIn[1] = OaFnMatrix::Scale(OaFnMatrix::Mul(aDivB, dOutDivB), -1.0f);
		}
	}
};

class OaGradNeg final : public OaGradNode {
public:
	void Backward(const OaMatrix& InDOut, OaVec<OaMatrix>& OutDIn) override {
		if (OutDIn.Size() > 0) OutDIn[0] = OaFnMatrix::Scale(InDOut, -1.0f);
	}
};

// ─── Broadcasting binary ops (Add/Sub/Mul/Div with shape mismatch) ──────────
// The element-wise *.gen.h nodes assume both operands share the output shape.
// When one operand was broadcast (e.g. NormWeight [1,1,H,P] * y [B,L,H,P], or a
// per-channel bias add), the upstream grad has the OUTPUT shape and must be
// summed back down to each operand's original shape. Without this reduction the
// engine drops the contribution entirely (numel mismatch) → the broadcast
// operand AND everything upstream of it silently receive ZERO gradient. This is
// the canonical broadcast-backprop reduction.
enum class OaBcastBinOp { Add, Sub, Mul, Div };

class OaGradBcastBinary final : public OaGradNode {
public:
	OaBcastBinOp Op_ = OaBcastBinOp::Mul;

	// Sum `g` (output-shaped) down to `target` by reducing every axis the target
	// broadcast over (target dim == 1 while g dim > 1, plus any extra leading axes).
	static OaMatrix SumToShape(const OaMatrix& InG, const OaMatrixShape& InTarget) {
		OaMatrixShape gs = InG.GetShape();
		if (gs == InTarget) return InG;
		OaMatrix cur = InG;
		OaMatrixShape cs = gs;
		const OaI32 lead = gs.Rank - InTarget.Rank;  // extra leading axes on g
		for (OaI32 axis = 0; axis < gs.Rank; ++axis) {
			const OaI32 taxis = axis - lead;          // matching target axis (left-padded)
			const OaI64 tdim = (taxis >= 0) ? InTarget.Dims[taxis] : 1;
			if (tdim == 1 && cs.Dims[axis] > 1) {
				cur = OaFnMatrix::Sum(cur, axis);     // keepdim → size 1 on `axis`
				cs.Dims[axis] = 1;
			}
		}
		return (cur.GetShape() == InTarget) ? cur : cur.Reshape(InTarget);
	}

	void Backward(const OaMatrix& InDOut, OaVec<OaMatrix>& OutDIn) override {
		const OaMatrix& a = Saved_[0];
		const OaMatrix& b = Saved_[1];
		OaMatrix gA, gB;
		switch (Op_) {
			case OaBcastBinOp::Add:
				gA = InDOut;
				gB = InDOut;
				break;
			case OaBcastBinOp::Sub:
				gA = InDOut;
				gB = OaFnMatrix::Scale(InDOut, -1.0f);
				break;
			case OaBcastBinOp::Mul:
				gA = OaFnMatrix::Mul(InDOut, b);
				gB = OaFnMatrix::Mul(InDOut, a);
				break;
			case OaBcastBinOp::Div:
				// d/da (a/b) = 1/b ; d/db (a/b) = -a/b^2
				gA = OaFnMatrix::Div(InDOut, b);
				gB = OaFnMatrix::Scale(OaFnMatrix::Div(OaFnMatrix::Mul(InDOut, a),
					OaFnMatrix::Mul(b, b)), -1.0f);
				break;
		}
		if (OutDIn.Size() > 0) OutDIn[0] = SumToShape(gA, a.GetShape());
		if (OutDIn.Size() > 1) OutDIn[1] = SumToShape(gB, b.GetShape());
	}
};

// ─── Sum reduction ──────────────────────────────────────────────────────────
// d/dx of sum is 1 for every summed element, so the backward simply broadcasts
// (expands) the upstream grad back to the input shape: each input element that
// fed a given output element receives that output element's upstream grad.
// Works for both keepdim axis reductions (out has size 1 on the reduced axis)
// and full reductions (out is scalar [1]). Without this node, OaFnMatrix::Sum
// was a hard stop for autograd — any loss that reduced with Sum (e.g. MSE,
// velocity/contact losses) silently produced ZERO gradient everywhere.
class OaGradSum final : public OaGradNode {
public:
	void Backward(const OaMatrix& InDOut, OaVec<OaMatrix>& OutDIn) override {
		if (OutDIn.Size() == 0) return;
		const OaMatrix& x = Saved_[0];
		auto ones = OaFnMatrix::Ones(x.GetShape(), x.GetDtype());
		OutDIn[0] = OaFnMatrix::Mul(ones, InDOut);  // broadcast-expand to input shape
	}
};

class OaGradMean final : public OaGradNode {
public:
	OaI32 Dim_ = -1;
	void Backward(const OaMatrix& InDOut, OaVec<OaMatrix>& OutDIn) override {
		if (OutDIn.Size() == 0) return;
		const OaMatrix& input = Saved_[0];
		const OaI64 count = Dim_ >= 0 and Dim_ < input.Rank()
			? input.Size(Dim_) : input.NumElements();
		auto scale = OaFnMatrix::Full(
			input.GetShape(), 1.0 / static_cast<OaF64>(count), input.GetDtype());
		OutDIn[0] = OaFnMatrix::Mul(scale, InDOut);
	}
};

class OaGradAbs final : public OaGradNode {
public:
	void Backward(const OaMatrix& InDOut, OaVec<OaMatrix>& OutDIn) override {
		if (OutDIn.Size() > 0) {
			const OaMatrix& x = Saved_[0];
			auto posMask = OaFnMatrix::Equal(OaFnMatrix::ClampMin(x, 0.0f), 0.0f);
			auto negMask = OaFnMatrix::Equal(OaFnMatrix::ClampMax(x, 0.0f), 0.0f);
			auto sign = OaFnMatrix::Sub(posMask, negMask);
			OutDIn[0] = OaFnMatrix::Mul(InDOut, sign);
		}
	}
};

class OaGradClampMax final : public OaGradNode {
public:
	explicit OaGradClampMax(OaF32 InMax) noexcept : Max_(InMax) {}
	void Backward(const OaMatrix& InDOut, OaVec<OaMatrix>& OutDIn) override {
		if (OutDIn.Size() > 0) {
			const OaMatrix& x = Saved_[0];
			auto diff = OaFnMatrix::SubScalar(x, Max_);
			auto mask = OaFnMatrix::Equal(OaFnMatrix::ClampMax(diff, 0.0f), 0.0f);
			OutDIn[0] = OaFnMatrix::Mul(InDOut, mask);
		}
	}
private:
	OaF32 Max_;
};

class OaGradClampMin final : public OaGradNode {
public:
	explicit OaGradClampMin(OaF32 InMin) noexcept : Min_(InMin) {}
	void Backward(const OaMatrix& InDOut, OaVec<OaMatrix>& OutDIn) override {
		if (OutDIn.Size() > 0) {
			const OaMatrix& x = Saved_[0];
			auto negX = OaFnMatrix::Scale(x, -1.0f);
			auto diff = OaFnMatrix::AddScalar(negX, Min_);
			auto mask = OaFnMatrix::Equal(OaFnMatrix::ClampMax(diff, 0.0f), 0.0f);
			OutDIn[0] = OaFnMatrix::Mul(InDOut, mask);
		}
	}
private:
	OaF32 Min_;
};

// ─── Activation manual (complex or no schema entry) ─────────────────────────

class OaGradSoftmax final : public OaGradNode {
public:
	OaI32 Dim_ = -1;
	void Backward(const OaMatrix& InDOut, OaVec<OaMatrix>& OutDIn) override {
		if (OutDIn.Size() > 0) {
			OutDIn[0] = OaFnMatrix::SoftmaxBwd(Saved_[0], InDOut, Dim_);
		}
	}
};

class OaGradLogSoftmax final : public OaGradNode {
public:
	OaI32 Dim_ = -1;
	void Backward(const OaMatrix& InDOut, OaVec<OaMatrix>& OutDIn) override {
		if (OutDIn.Size() > 0) {
			OutDIn[0] = OaFnMatrix::LogSoftmaxBwd(Saved_[0], InDOut, Dim_);
		}
	}
};

class OaGradSoftmaxScaledMasked final : public OaGradNode {
public:
	OaF32 Scale_ = 1.0F;

	void Backward(const OaMatrix& InDOut, OaVec<OaMatrix>& OutDIn) override {
		if (OutDIn.Size() > 0) {
			OutDIn[0] = OaFnMatrix::SoftmaxScaledMaskedBwd(Saved_[0], InDOut, Scale_);
		}
	}
};

class OaGradSiluMul final : public OaGradNode {
public:
	void Backward(const OaMatrix& InDOut, OaVec<OaMatrix>& OutDIn) override {
		if (OutDIn.Size() > 0) OutDIn[0] = OaFnMatrix::SiluMulBwd(Saved_[0], InDOut);
	}
};

class OaGradGeglu final : public OaGradNode {
public:
	void Backward(const OaMatrix& InDOut, OaVec<OaMatrix>& OutDIn) override {
		if (OutDIn.Size() > 0) OutDIn[0] = OaFnMatrix::GegluBwd(Saved_[0], InDOut);
	}
};

class OaGradMax final : public OaGradNode {
public:
	// Saved_[0] = input X, Saved_[1] = max scalar (forward output).
	// Routes the upstream scalar grad to the element(s) equal to the max.
	void Backward(const OaMatrix& InDOut, OaVec<OaMatrix>& OutDIn) override {
		if (OutDIn.Size() > 0) OutDIn[0] = OaFnMatrix::MaxBwd(Saved_[0], Saved_[1], InDOut);
	}
};

// ─── Linear / BLAS ────────────────────────────────────────────────────────

class OaGradLinear final : public OaGradNode {
public:
	void Backward(const OaMatrix& InDOut, OaVec<OaMatrix>& OutDIn) override {
		const OaMatrix& x      = Saved_[0];
		const OaMatrix& weight = Saved_[1];
		const bool needDX = OutDIn.Size() > 0;
		const bool needDW = OutDIn.Size() > 1;
		const bool needDB = OutDIn.Size() > 2;
		if (needDX) OutDIn[0] = OaFnMatrix::LinearDataBwd(InDOut, weight);
		if (needDW or needDB) {
			auto gwb = OaFnMatrix::LinearWeightBiasBwd(x, InDOut);
			if (needDW) OutDIn[1] = gwb.GradWeight;
			if (needDB) OutDIn[2] = gwb.GradBias;
		}
	}
};

class OaGradPackedLinear2 final : public OaGradNode {
public:
	OaI64 N0_ = 0, N1_ = 0;
	bool HasBias_ = false;
	void Backward(const OaMatrix& InDOut, OaVec<OaMatrix>& OutDIn) override {
		const auto d0 = OaFnMatrix::Slice(InDOut, 1, 0, N0_);
		const auto d1 = OaFnMatrix::Slice(InDOut, 1, N0_, N0_ + N1_);
		if (OutDIn.Size() > 0) OutDIn[0] = OaFnMatrix::Add(
			OaFnMatrix::LinearDataBwd(d0, Saved_[1]),
			OaFnMatrix::LinearDataBwd(d1, Saved_[2]));
		if (OutDIn.Size() > 1) {
			auto g0 = OaFnMatrix::LinearWeightBiasBwd(Saved_[0], d0);
			auto g1 = OaFnMatrix::LinearWeightBiasBwd(Saved_[0], d1);
			OutDIn[1] = g0.GradWeight;
			if (OutDIn.Size() > 2) OutDIn[2] = g1.GradWeight;
			if (HasBias_ and OutDIn.Size() > 4) {
				OutDIn[3] = g0.GradBias; OutDIn[4] = g1.GradBias;
			}
		}
	}
};

class OaGradPackedLinear3 final : public OaGradNode {
public:
	OaI64 N0_ = 0, N1_ = 0, N2_ = 0;
	bool HasBias_ = false;
	void Backward(const OaMatrix& InDOut, OaVec<OaMatrix>& OutDIn) override {
		const auto d0 = OaFnMatrix::Slice(InDOut, 1, 0, N0_);
		const auto d1 = OaFnMatrix::Slice(InDOut, 1, N0_, N0_ + N1_);
		const auto d2 = OaFnMatrix::Slice(InDOut, 1, N0_ + N1_, N0_ + N1_ + N2_);
		if (OutDIn.Size() > 0) OutDIn[0] = OaFnMatrix::Add(OaFnMatrix::Add(
			OaFnMatrix::LinearDataBwd(d0, Saved_[1]),
			OaFnMatrix::LinearDataBwd(d1, Saved_[2])),
			OaFnMatrix::LinearDataBwd(d2, Saved_[3]));
		if (OutDIn.Size() > 1) {
			auto g0 = OaFnMatrix::LinearWeightBiasBwd(Saved_[0], d0);
			auto g1 = OaFnMatrix::LinearWeightBiasBwd(Saved_[0], d1);
			auto g2 = OaFnMatrix::LinearWeightBiasBwd(Saved_[0], d2);
			OutDIn[1] = g0.GradWeight;
			if (OutDIn.Size() > 2) OutDIn[2] = g1.GradWeight;
			if (OutDIn.Size() > 3) OutDIn[3] = g2.GradWeight;
			if (HasBias_ and OutDIn.Size() > 6) {
				OutDIn[4] = g0.GradBias; OutDIn[5] = g1.GradBias; OutDIn[6] = g2.GradBias;
			}
		}
	}
};

class OaGradLinearRelu final : public OaGradNode {
public:
	void Backward(const OaMatrix& InDOut, OaVec<OaMatrix>& OutDIn) override {
		const OaMatrix& x      = Saved_[0];
		const OaMatrix& weight = Saved_[1];
		const OaMatrix& act    = Saved_[2];
		const bool needDX = OutDIn.Size() > 0;
		const bool needDW = OutDIn.Size() > 1;
		const bool needDB = OutDIn.Size() > 2;
		auto dZ = OaFnMatrix::ReluBwd(act, InDOut);
		if (needDX) OutDIn[0] = OaFnMatrix::LinearDataBwd(dZ, weight);
		if (needDW or needDB) {
			auto gwb = OaFnMatrix::LinearWeightBiasBwd(x, dZ);
			if (needDW) OutDIn[1] = gwb.GradWeight;
			if (needDB) OutDIn[2] = gwb.GradBias;
		}
	}
};

// LinearGelu fused backward. Unlike ReLU (whose mask is recoverable from the
// forward output), GELU'(pre) is a function of the PRE-activation, which the
// fused forward kernel discards. So recompute pre = X @ W^T + bias (one GEMM)
// here, then dZ = GeluBwd(pre, dOut). The forward still saved one dispatch; the
// only training cost is this single recompute. Saved_ = {x, weight, bias}.
class OaGradLinearGelu final : public OaGradNode {
public:
	void Backward(const OaMatrix& InDOut, OaVec<OaMatrix>& OutDIn) override {
		const OaMatrix& x      = Saved_[0];
		const OaMatrix& weight = Saved_[1];
		const OaMatrix& bias   = Saved_[2];
		const bool needDX = OutDIn.Size() > 0;
		const bool needDW = OutDIn.Size() > 1;
		const bool needDB = OutDIn.Size() > 2;
		auto pre = OaFnMatrix::Linear(x, weight, bias);
		auto dZ  = OaFnMatrix::GeluBwd(pre, InDOut);
		if (needDX) OutDIn[0] = OaFnMatrix::LinearDataBwd(dZ, weight);
		if (needDW or needDB) {
			auto gwb = OaFnMatrix::LinearWeightBiasBwd(x, dZ);
			if (needDW) OutDIn[1] = gwb.GradWeight;
			if (needDB) OutDIn[2] = gwb.GradBias;
		}
	}
};

class OaGradMatMulNt final : public OaGradNode {
public:
	void Backward(const OaMatrix& InDOut, OaVec<OaMatrix>& OutDIn) override {
		const OaMatrix& A = Saved_[0];
		const OaMatrix& B = Saved_[1];
		// Normalize InDOut: callers (e.g. Mamba3 returning a rank-3 view of the 2D matmul
		// result, or LM code) may deliver dout via shared autograd meta with viewed shape.
		// Forward: out[m,n] = sum_k A[m,k] * B[n,k]  (B is [N,K]; equivalent to A @ B^T).
		// dA = dOut @ B,  dB = dOut^T @ A.
		OaMatrixShape expected{ A.Size(0), B.Size(0) };
		OaMatrix dOut2 = InDOut;
		if (InDOut.GetShape() != expected && InDOut.NumElements() == expected.NumElements()) {
			dOut2 = InDOut.Reshape(expected);
		}
		// Use Linear*Bwd kernels — MatMul(Transpose…) during backward produces zeros on GPU.
		if (OutDIn.Size() > 0) OutDIn[0] = OaFnMatrix::LinearDataBwd(dOut2, B);
		if (OutDIn.Size() > 1) OutDIn[1] = OaFnMatrix::LinearWeightBwd(A, dOut2);
	}
};

class OaGradTranspose final : public OaGradNode {
public:
	OaI32 Dim0_ = 0;
	OaI32 Dim1_ = 1;
	void Backward(const OaMatrix& InDOut, OaVec<OaMatrix>& OutDIn) override {
		if (OutDIn.Size() > 0) OutDIn[0] = OaFnMatrix::Transpose(InDOut, Dim0_, Dim1_);
	}
};

class OaGradReshape final : public OaGradNode {
public:
	OaMatrixShape InputShape_{};
	void Backward(const OaMatrix& InDOut, OaVec<OaMatrix>& OutDIn) override {
		if (OutDIn.Size() > 0) OutDIn[0] = InDOut.Reshape(InputShape_);
	}
};

// Differentiable dtype cast (mixed-precision boundary). Forward casts src→dst;
// the upstream grad arrives in the dst dtype, so backward casts it back to the
// src dtype. This is the primitive that lets fp32-only ops (SSM scans, etc.) be
// wrapped as fp32 islands inside a bf16 graph with gradients flowing through.
class OaGradCast final : public OaGradNode {
public:
	OaScalarType SrcDtype_ = OaScalarType::Float32;
	void Backward(const OaMatrix& InDOut, OaVec<OaMatrix>& OutDIn) override {
		if (OutDIn.Size() > 0) OutDIn[0] = OaFnMatrix::Cast(InDOut, SrcDtype_);
	}
};

class OaGradRepeatInterleave final : public OaGradNode {
public:
	OaI32 Repeats_ = 1;
	OaI32 Dim_ = 0;
	void Backward(const OaMatrix& InDOut, OaVec<OaMatrix>& OutDIn) override {
		if (OutDIn.Size() > 0) {
			const OaMatrix& input = Saved_[0];
			OutDIn[0] = OaFnMatrix::RepeatInterleaveBwd(InDOut, input.GetShape(), Repeats_, Dim_);
		}
	}
};

class OaGradCausalMask final : public OaGradNode {
public:
	void Backward(const OaMatrix& InDOut, OaVec<OaMatrix>& OutDIn) override {
		if (OutDIn.Size() > 0) OutDIn[0] = OaFnMatrix::CausalMaskBwd(InDOut);
	}
};

class OaGradCompactRows final : public OaGradNode {
public:
	OaMatrix RowMap_;
	OaMatrix Count_;
	OaMatrix DispatchArgs_;
	void Backward(const OaMatrix& InDOut, OaVec<OaMatrix>& OutDIn) override {
		if (OutDIn.Size() > 0) {
			const OaMatrix& input = Saved_[0];
			OutDIn[0] = OaFnMatrix::CompactRowsBwd(
				InDOut, RowMap_, Count_, DispatchArgs_, input.GetShape());
		}
	}
};

class OaGradScatterRows final : public OaGradNode {
public:
	OaMatrix RowMap_;
	OaMatrix Count_;
	OaMatrix DispatchArgs_;
	void Backward(const OaMatrix& InDOut, OaVec<OaMatrix>& OutDIn) override {
		if (OutDIn.Size() < 2) return;
		OutDIn[0] = OaFnMatrix::Copy(InDOut);
		OutDIn[1] = DispatchArgs_.NumElements() == 3
			? OaFnMatrix::ScatterRowsBwdSource(InDOut, RowMap_, Count_, DispatchArgs_)
			: OaFnMatrix::ScatterRowsBwdSource(InDOut, RowMap_, Count_);
	}
};

// ─── Convolution ──────────────────────────────────────────────────────────

class OaGradConv2d final : public OaGradNode {
public:
	OaU32 Stride_ = 1;
	OaU32 Padding_ = 0;
	OaU32 Groups_ = 1;
	void Backward(const OaMatrix& InDOut, OaVec<OaMatrix>& OutDIn) override {
		const OaMatrix& x = Saved_[0];
		const OaMatrix& weight = Saved_[1];
		const OaMatrixShape input_shape = x.GetShape();
		const bool needDX = OutDIn.Size() > 0;
		const bool needDW = OutDIn.Size() > 1;
		const bool needDB = OutDIn.Size() > 2;
		if (needDX) OutDIn[0] = OaFnMatrix::Conv2dBwdData(InDOut, weight, Stride_, Padding_, input_shape, Groups_);
		if (needDW or needDB) {
			auto gwb = OaFnMatrix::Conv2dBwdWeight(x, InDOut, weight, Stride_, Padding_, Groups_);
			if (needDW) OutDIn[1] = gwb.GradWeight;
			if (needDB) OutDIn[2] = gwb.GradBias;
		}
	}
};

// OaGradConv1d retired: OaConv1d::Forward now uses OaFnMatrix::Conv1dGemm, which
// composes its own backward from Im2Col1d + MatMulNt + BiasAdd. The bare scalar
// Conv1d forward kernel it depended on was retired. Conv1dBwdData/Conv1dBwdWeight
// survive — they back OaConvTranspose1d (below) and are the adjoints reused there.

// OaGradConvTranspose2d — 2D transposed convolution gradient node.
// Forward: y = ConvTranspose2d(x, W, b) with W [InC, OutC, K, K].
// Backward reuses Conv2d kernels: dX = Conv2d(dOut, W), dW/dB = ConvTranspose2dBwdWeight.
class OaGradConvTranspose2d final : public OaGradNode {
public:
	OaU32 Stride_ = 1;
	OaU32 Padding_ = 0;
	void Backward(const OaMatrix& InDOut, OaVec<OaMatrix>& OutDIn) override {
		const OaMatrix& x = Saved_[0];
		const OaMatrix& weight = Saved_[1];
		const OaMatrixShape input_shape = x.GetShape();
		const bool needDX = OutDIn.Size() > 0;
		const bool needDW = OutDIn.Size() > 1;
		const bool needDB = OutDIn.Size() > 2;
		if (needDX) OutDIn[0] = OaFnMatrix::ConvTranspose2dBwdData(InDOut, weight, Stride_, Padding_, input_shape);
		if (needDW or needDB) {
			auto gwb = OaFnMatrix::ConvTranspose2dBwdWeight(x, InDOut, weight, Stride_, Padding_);
			if (needDW) OutDIn[1] = gwb.GradWeight;
			if (needDB) OutDIn[2] = gwb.GradBias;
		}
	}
};

// OaConvTranspose1d is the ADJOINT of Conv1d (learnable upsampling), so its forward is
// exactly Conv1d's backward-data and its backward reuses Conv1d's forward + weight-grad —
// no new kernels. Weight is [InCh, OutCh, K] (PyTorch ConvTranspose1d convention; equals
// the underlying conv's [Cout, Cin, K] with Cout=InCh, Cin=OutCh). No bias (v1).
//   forward: y  = Conv1dBwdData(x, W)                  [B,InCh,Lin] → [B,OutCh,Lout]
//   dX = Conv1d(dOut, W)                (adjoint of bwd-data is the forward conv)
//   dW = Conv1dBwdWeight(input=dOut, dOut=x, W).GradWeight
class OaGradConvTranspose1d final : public OaGradNode {
public:
	OaU32 Stride_ = 1;
	OaU32 Padding_ = 0;
	void Backward(const OaMatrix& InDOut, OaVec<OaMatrix>& OutDIn) override {
		const OaMatrix& x      = Saved_[0];
		const OaMatrix& weight = Saved_[1];
		const bool needDX = OutDIn.Size() > 0;
		const bool needDW = OutDIn.Size() > 1;
		if (needDX) {
			// Adjoint of ConvTranspose1d's bwd-data forward is a plain conv. Runs
			// under autograd-disabled backward, so Conv1dGemm here is pure im2col+GEMM
			// math (its sub-ops no-op their grad nodes) — the retired scalar Conv1d.
			OaMatrix zeroBias = OaFnMatrix::Zeros(OaMatrixShape{weight.Size(0)}, weight.GetDtype());
			OutDIn[0] = OaFnMatrix::Conv1dGemm(InDOut, weight, zeroBias,
				static_cast<OaI32>(Stride_), static_cast<OaI32>(Padding_), 1);
		}
		if (needDW) {
			auto gwb = OaFnMatrix::Conv1dBwdWeight(InDOut, x, weight, Stride_, Padding_, 1);
			OutDIn[1] = gwb.GradWeight;
		}
	}
};

// ─── Embedding / Gather ─────────────────────────────────────────────────────

class OaGradGather final : public OaGradNode {
public:
	void Backward(const OaMatrix& InDOut, OaVec<OaMatrix>& OutDIn) override {
		const OaMatrix& indices = Saved_[0];
		const OaMatrix& weight  = Saved_[1];
		const bool needDW = OutDIn.Size() > 0;
		if (needDW) {
			OaI32 vocabSize = static_cast<OaI32>(weight.Size(0));
			OaI32 embedDim  = static_cast<OaI32>(weight.Size(1));
			OutDIn[0] = OaFnMatrix::GatherBwd(indices, InDOut, vocabSize, embedDim);
		}
	}
};

class OaGradGatherLastDim final : public OaGradNode {
public:
	void Backward(const OaMatrix& InDOut, OaVec<OaMatrix>& OutDIn) override {
		if (OutDIn.Size() == 0) return;
		OutDIn[0] = OaFnMatrix::GatherLastDimBwd(
			InDOut, Saved_[1], static_cast<OaI32>(Saved_[0].Size(1)));
	}
};

class OaGradMoeRouteWeights final : public OaGradNode {
public:
	void Backward(const OaMatrix& InDOut, OaVec<OaMatrix>& OutDIn) override {
		if (OutDIn.Size() == 0) return;
		OutDIn[0] = OaFnMatrix::MoeRouteWeightsBwd(
			InDOut, Saved_[0], Saved_[1], Saved_[2]);
	}
};

class OaGradGroupedGemmM final : public OaGradNode {
public:
	void Backward(const OaMatrix& InDOut, OaVec<OaMatrix>& OutDIn) override {
		if (OutDIn.Size() < 2) return;
		auto result = OaFnMatrix::GroupedGemmMBwd(
			InDOut, Saved_[0], Saved_[1], Saved_[2]);
		OutDIn[0] = result.DInput;
		OutDIn[1] = result.DWeight;
	}
};

class OaGradGroupedLinearM final : public OaGradNode {
public:
	void Backward(const OaMatrix& InDOut, OaVec<OaMatrix>& OutDIn) override {
		if (OutDIn.Size() < 3) return;
		auto result = OaFnMatrix::GroupedLinearMBwd(
			InDOut, Saved_[0], Saved_[1], Saved_[3]);
		OutDIn[0] = result.DInput;
		OutDIn[1] = result.DWeight;
		OutDIn[2] = result.DBias;
	}
};

class OaGradMoeCombine final : public OaGradNode {
public:
	void Backward(const OaMatrix& InDOut, OaVec<OaMatrix>& OutDIn) override {
		if (OutDIn.Size() < 2) return;
		auto result = OaFnMatrix::MoeCombineBwd(
			InDOut, Saved_[0], Saved_[1], Saved_[2], Saved_[3]);
		OutDIn[0] = result.DPacked;
		OutDIn[1] = result.DRouteGate;
	}
};

class OaGradMoeGather final : public OaGradNode {
public:
	void Backward(const OaMatrix& InDOut, OaVec<OaMatrix>& OutDIn) override {
		if (OutDIn.Size() == 0) return;
		const OaMatrix& input = Saved_[0];
		OutDIn[0] = OaFnMatrix::MoeGatherBwd(
			InDOut, Saved_[2], static_cast<OaI32>(input.Size(0)));
	}
};

class OaGradScatterAddRows final : public OaGradNode {
public:
	void Backward(const OaMatrix& InDOut, OaVec<OaMatrix>& OutDIn) override {
		if (OutDIn.Size() > 0) OutDIn[0] = OaFnMatrix::Gather(InDOut, Saved_[0]);
	}
};

// OaGradBmm — batched matmul out[n] = A[n] @ B[n]. Gradients are batched matmuls
// with the other operand transposed (per batch): dA = dOut @ Bᵀ, dB = Aᵀ @ dOut.
// Reuses the batched 3D transpose (Transpose on axes 1,2).
class OaGradBmm final : public OaGradNode {
public:
	void Backward(const OaMatrix& InDOut, OaVec<OaMatrix>& OutDIn) override {
		const OaMatrix& a = Saved_[0];   // [N, M, K]
		const OaMatrix& b = Saved_[1];   // [N, K, P]
		if (OutDIn.Size() > 0)           // dA = dOut[N,M,P] @ Bᵀ[N,P,K]
			OutDIn[0] = OaFnMatrix::Bmm(InDOut, OaFnMatrix::Transpose(b, 1, 2));
		if (OutDIn.Size() > 1)           // dB = Aᵀ[N,K,M] @ dOut[N,M,P]
			OutDIn[1] = OaFnMatrix::Bmm(OaFnMatrix::Transpose(a, 1, 2), InDOut);
	}
};

// ─── Gated / Composite ──────────────────────────────────────────────────────

class OaGradSwiglu final : public OaGradNode {
public:
	void Backward(const OaMatrix& InDOut, OaVec<OaMatrix>& OutDIn) override {
		const OaMatrix& gate = Saved_[0];
		const OaMatrix& up   = Saved_[1];
		const OaMatrix& out  = Saved_[2];
		const bool needGate = OutDIn.Size() > 0;
		const bool needUp   = OutDIn.Size() > 1;
		if (needGate or needUp) {
			auto grads = OaFnMatrix::SwigluBwd(gate, up, out, InDOut);
			if (needGate) OutDIn[0] = grads.GateGrad;
			if (needUp)   OutDIn[1] = grads.UpGrad;
		}
	}
};

class OaGradGruCellPointwise final : public OaGradNode {
public:
	OaI32 HiddenSize_ = 0;
	OaU32 TimeOffset_ = 0;
	OaU32 BatchStride_ = 1;
	
	void Backward(const OaMatrix& InDOut, OaVec<OaMatrix>& OutDIn) override {
		const OaMatrix& gatesI = Saved_[0];
		const OaMatrix& gatesH = Saved_[1];
		const OaMatrix& hidden = Saved_[2];
		const bool needDGatesI = OutDIn.Size() > 0;
		const bool needDGatesH = OutDIn.Size() > 1;
		const bool needDHidden = OutDIn.Size() > 2;
		if (needDGatesI or needDGatesH or needDHidden) {
			auto grads = OaFnMatrix::GruCellPointwiseBwd(gatesI, gatesH, hidden, InDOut, HiddenSize_, TimeOffset_, BatchStride_);
			if (needDGatesI) OutDIn[0] = grads.DGatesI;
			if (needDGatesH) OutDIn[1] = grads.DGatesH;
			if (needDHidden) OutDIn[2] = grads.DHidden;
		}
	}
};

// Whole-sequence GRU scan backward. Saved_ layout (always 4 slots):
//   [0] gatesIAll [B*S,3H], [1] W_hh [3H,H], [2] biasHh (dummy if !HasBias_), [3] hprevAll [B,S,H].
// Graph inputs = (gatesIAll, W_hh, biasHh?). The BPTT recurrence scan (GruScanBwd)
// produces dGatesI and dGatesH; the recurrent weight/bias grad is one
// LinearWeightBiasBwd(hprevAll, dGatesH) (sums over all B*S rows = over t and b).
class OaGradGruScan final : public OaGradNode {
public:
	OaI32 HiddenSize_ = 0;
	OaI32 SeqLen_ = 0;
	OaI32 Batch_ = 0;
	bool HasBias_ = false;

	void Backward(const OaMatrix& InDOut, OaVec<OaMatrix>& OutDIn) override {
		const OaMatrix& gatesI = Saved_[0];
		const OaMatrix& wHh    = Saved_[1];
		const OaMatrix& hprev3d = Saved_[3];
		const bool needDGatesI = OutDIn.Size() > 0;
		const bool needDW      = OutDIn.Size() > 1;
		const bool needDBias   = OutDIn.Size() > 2;
		if (not (needDGatesI or needDW or needDBias)) return;
		const OaMatrix* biasHh = HasBias_ ? &Saved_[2] : nullptr;
		auto grads = OaFnMatrix::GruScanBwd(InDOut, gatesI, hprev3d, wHh, biasHh, HiddenSize_, SeqLen_, Batch_);
		if (needDGatesI) OutDIn[0] = grads.DGatesI;
		if (needDW or needDBias) {
			// LinearWeightBiasBwd expects 2D [batch, in_features]; hprev is stored as [B,S,H]
			// but memory layout is identical to [B*S, H].
			auto hprev2d = hprev3d.Reshape(OaMatrixShape{static_cast<OaI64>(Batch_) * SeqLen_, HiddenSize_});
			auto gwb = OaFnMatrix::LinearWeightBiasBwd(hprev2d, grads.DGatesH);
			if (needDW) OutDIn[1] = gwb.GradWeight;
			if (needDBias) OutDIn[2] = gwb.GradBias;
		}
	}
};

// Fused vanilla-RNN pointwise: h_new = tanh(gates_i + gates_h).
// Graph inputs are (gates_i, gates_h); both gradients equal dL/da, computed by
// the fused backward kernel from the saved gate projections. gates_i is the whole-
// sequence [B*T, H] projection, so the backward scatters DGatesI into this
// timestep's rows (TimeOffset_/BatchStride_) — mirrors OaGradGruCellPointwise.
class OaGradRnnCellPointwise final : public OaGradNode {
public:
	OaI32 HiddenSize_ = 0;
	OaU32 TimeOffset_ = 0;
	OaU32 BatchStride_ = 1;

	void Backward(const OaMatrix& InDOut, OaVec<OaMatrix>& OutDIn) override {
		const OaMatrix& gatesI = Saved_[0];
		const OaMatrix& gatesH = Saved_[1];
		const bool needDGatesI = OutDIn.Size() > 0;
		const bool needDGatesH = OutDIn.Size() > 1;
		if (needDGatesI or needDGatesH) {
			auto grads = OaFnMatrix::RnnCellPointwiseBwd(gatesI, gatesH, InDOut, HiddenSize_, TimeOffset_, BatchStride_);
			if (needDGatesI) OutDIn[0] = grads.DGatesI;
			if (needDGatesH) OutDIn[1] = grads.DGatesH;
		}
	}
};

// Whole-sequence RNN recurrent scan: identical math to RnnCellLinear but in one
// dispatch. Graph inputs are (gates_i, w_hh, b_hh optional). Backward calls the
// fused RnnScanBwd kernel and LinearWeightBiasBwd for the recurrent weight/bias.
class OaGradRnnScan final : public OaGradNode {
public:
	OaI32 HiddenSize_ = 0;
	OaI32 SeqLen_ = 0;
	OaI32 Batch_ = 0;
	bool HasBias_ = false;

	void Backward(const OaMatrix& InDOut, OaVec<OaMatrix>& OutDIn) override {
		const OaMatrix& gatesI = Saved_[0];
		const OaMatrix& wHh    = Saved_[1];
		const OaMatrix& hprev3d = Saved_[3];
		const bool needDGatesI = OutDIn.Size() > 0;
		const bool needDW      = OutDIn.Size() > 1;
		const bool needDBias   = OutDIn.Size() > 2;
		if (not (needDGatesI or needDW or needDBias)) return;
		const OaMatrix* biasHh = HasBias_ ? &Saved_[2] : nullptr;
		auto grads = OaFnMatrix::RnnScanBwd(InDOut, gatesI, hprev3d, wHh, biasHh, HiddenSize_, SeqLen_, Batch_);
		if (needDGatesI) OutDIn[0] = grads.DGatesI;
		if (needDW or needDBias) {
			// LinearWeightBiasBwd expects 2D [batch, in_features]; hprev is stored as [B,S,H]
			// but memory layout is identical to [B*S, H].
			auto hprev2d = hprev3d.Reshape(OaMatrixShape{static_cast<OaI64>(Batch_) * SeqLen_, HiddenSize_});
			auto gwb = OaFnMatrix::LinearWeightBiasBwd(hprev2d, grads.DGatesH);
			if (needDW) OutDIn[1] = gwb.GradWeight;
			if (needDBias) OutDIn[2] = gwb.GradBias;
		}
	}
};

// ─── Normalization ────────────────────────────────────────────────────────

class OaGradLayerNorm final : public OaGradNode {
public:
	OaF32 Eps_ = 1e-5F;
	void Backward(const OaMatrix& InDOut, OaVec<OaMatrix>& OutDIn) override {
		const OaMatrix& x = Saved_[0];
		const OaMatrix& weight = Saved_[1];
		const OaMatrix& bias   = Saved_[2];
		const OaMatrix& out    = Saved_[3];
		const bool needDX = OutDIn.Size() > 0;
		const bool needDW = OutDIn.Size() > 1;
		const bool needDB = OutDIn.Size() > 2;
		if (needDX or needDW or needDB) {
			auto grads = OaFnMatrix::LayerNormBwd(
				x, weight, bias, out, out, out, InDOut, Eps_);
			if (needDX) OutDIn[0] = grads.DX;
			if (needDW) OutDIn[1] = grads.DWeight;
			if (needDB) OutDIn[2] = grads.DBias;
		}
	}
};

class OaGradRmsNorm final : public OaGradNode {
public:
	OaF32 Eps_ = 1e-5F;
	void Backward(const OaMatrix& InDOut, OaVec<OaMatrix>& OutDIn) override {
		const OaMatrix& x = Saved_[0];
		const OaMatrix& weight = Saved_[1];
		const OaMatrix& out    = Saved_[2];
		const OaMatrix& rstd   = Saved_[3];
		const bool needDX = OutDIn.Size() > 0;
		const bool needDW = OutDIn.Size() > 1;
		if (needDX or needDW) {
			auto grads = OaFnMatrix::RmsNormBwd(
				x, weight, out, rstd, InDOut, Eps_);
			if (needDX) OutDIn[0] = grads.DX;
			if (needDW) OutDIn[1] = grads.DWeight;
		}
	}
};

class OaGradRmsNormGated final : public OaGradNode {
public:
	OaF32 Eps_ = 1e-5f;
	void Backward(const OaMatrix& InDOut, OaVec<OaMatrix>& OutDIn) override {
		const OaMatrix& x    = Saved_[0];
		const OaMatrix& w    = Saved_[1];
		const OaMatrix& bias = Saved_[2];
		const OaMatrix& z    = Saved_[3];
		auto g = OaFnMatrix::RmsNormGatedBwd(x, w, bias, z, InDOut, Eps_);
		if (OutDIn.Size() > 0) OutDIn[0] = g.DX;
		if (OutDIn.Size() > 1) OutDIn[1] = g.DWeight;
		if (OutDIn.Size() > 2) OutDIn[2] = g.DBias;
		if (OutDIn.Size() > 3) OutDIn[3] = g.DZ;
	}
};

// ─── Pooling ────────────────────────────────────────────────────────────────

class OaGradMaxPool2d final : public OaGradNode {
public:
	OaI32 KernelSize_ = 0;
	OaI32 Stride_ = 0;
	OaI32 Padding_ = 0;
	void Backward(const OaMatrix& InDOut, OaVec<OaMatrix>& OutDIn) override {
		// Saved_ = {InX, out, indices}; out is unused by the backward kernel.
		if (OutDIn.Size() > 0)
			OutDIn[0] = OaFnMatrix::MaxPool2dBwd(Saved_[0], Saved_[2], InDOut,
				KernelSize_, Stride_, Padding_);
	}
};

class OaGradAvgPool2d final : public OaGradNode {
public:
	OaI32 KernelSize_ = 0;
	OaI32 Stride_ = 0;
	OaI32 Padding_ = 0;
	
	void Backward(const OaMatrix& InDOut, OaVec<OaMatrix>& OutDIn) override {
		if (OutDIn.Size() > 0) {
			OutDIn[0] = OaFnMatrix::AvgPool2dBwd(Saved_[0], InDOut, KernelSize_, Stride_, Padding_);
		}
	}
};

// ─── Index / Shape ──────────────────────────────────────────────────────────

class OaGradConcat final : public OaGradNode {
public:
	OaI32 Dim_ = 0;
	OaVec<OaI64> Sizes_;
	void Backward(const OaMatrix& InDOut, OaVec<OaMatrix>& OutDIn) override {
		OaI64 offset = 0;
		for (OaI32 i = 0; i < static_cast<OaI32>(Sizes_.Size()); ++i) {
			if (i < static_cast<OaI32>(OutDIn.Size()) && Sizes_[i] > 0) {
				OutDIn[i] = OaFnMatrix::Slice(InDOut, Dim_, offset, offset + Sizes_[i]);
			}
			offset += Sizes_[i];
		}
	}
};

class OaGradSlice final : public OaGradNode {
public:
	OaMatrixShape InputShape_{};
	OaI32 Dim_ = 0;
	OaI64 Start_ = 0;
	OaI64 End_ = 0;
	void Backward(const OaMatrix& InDOut, OaVec<OaMatrix>& OutDIn) override {
		// Normalize InDOut to the slice's output shape (e.g. after downstream .Reshape
		// to flat 2D on a 3D slice result from the in_proj split). SliceBwd and the
		// copy region kernel expect the shape with the sliced dim.
		OaMatrixShape sliceShape = InputShape_;
		if (Dim_ >= 0 && Dim_ < sliceShape.Rank) {
			sliceShape[Dim_] = End_ - Start_;
		}
		OaMatrix dSlice = InDOut;
		if (InDOut.GetShape() != sliceShape && InDOut.NumElements() == sliceShape.NumElements()) {
			dSlice = InDOut.Reshape(sliceShape);
		}
		if (OutDIn.Size() > 0) OutDIn[0] = OaFnMatrix::SliceBwd(InputShape_, Dim_, Start_, End_, dSlice);
	}
};

class OaGradUpsample final : public OaGradNode {
public:
	OaI32 ScaleFactor_ = 1;
	bool IsBilinear_ = false;
	void Backward(const OaMatrix& InDOut, OaVec<OaMatrix>& OutDIn) override {
		if (OutDIn.Size() > 0) OutDIn[0] = OaFnMatrix::UpsampleBwd(Saved_[0], InDOut, ScaleFactor_, IsBilinear_);
	}
};

// ─── BatchNorm ──────────────────────────────────────────────────────────────

class OaGradBatchNorm2d final : public OaGradNode {
public:
	OaF32 Eps_ = 1e-5f;
	OaF32 Momentum_ = 0.1f;
	bool IsTraining_ = true;
	void Backward(const OaMatrix& InDOut, OaVec<OaMatrix>& OutDIn) override {
		const OaMatrix& x     = Saved_[0];
		const OaMatrix& gamma = Saved_[1];
		const OaMatrix& beta  = Saved_[2];
		const OaMatrix& mean  = Saved_[3];
		const OaMatrix& var   = Saved_[4];
		const OaMatrix& out   = Saved_[5];
		const bool needDX = OutDIn.Size() > 0;
		const bool needDG = OutDIn.Size() > 1;
		const bool needDB = OutDIn.Size() > 2;
		if (needDX or needDG or needDB) {
			auto grads = OaFnMatrix::BatchNorm2dBwd(x, gamma, beta, mean, var, out, InDOut, Eps_, IsTraining_);
			if (needDX) OutDIn[0] = grads.DX;
			if (needDG) OutDIn[1] = grads.DGamma;
			if (needDB) OutDIn[2] = grads.DBias;
		}
	}
};

// ─── Mamba-3 fused A·dt term ──────────────────────────────────────────────────
// Forward: ADT = min(-heavy_tail(ddA), -AFloor) * dt   (fused Mamba3Adt kernel).
// Both inputs are [B*S, H]; backward is pure elementwise (Mamba3AdtBwd kernel).
class OaGradMamba3Adt final : public OaGradNode {
public:
	explicit OaGradMamba3Adt(OaF32 InAFloor) noexcept : AFloor_(InAFloor) {}
	void Backward(const OaMatrix& InDOut, OaVec<OaMatrix>& OutDIn) override {
		const OaMatrix& ddA = Saved_[0];
		const OaMatrix& dt  = Saved_[1];
		auto g = OaFnMatrix::Mamba3AdtBwd(InDOut, ddA, dt, AFloor_);
		if (OutDIn.Size() > 0) OutDIn[0] = g.DDdA;
		if (OutDIn.Size() > 1) OutDIn[1] = g.DDt;
	}
private:
	OaF32 AFloor_;
};

// ─── Mamba-3 fused dt term ───────────────────────────────────────────────────
// Forward: DT = clamp(softplus(x), dt_min, dt_max)   (fused Mamba3Dt kernel).
// Backward is pure elementwise (Mamba3DtBwd kernel).
class OaGradMamba3Dt final : public OaGradNode {
public:
	OaGradMamba3Dt(OaF32 InDtMin, OaF32 InDtMax) noexcept : DtMin_(InDtMin), DtMax_(InDtMax) {}
	void Backward(const OaMatrix& InDOut, OaVec<OaMatrix>& OutDIn) override {
		const OaMatrix& x = Saved_[0];
		if (OutDIn.Size() > 0) OutDIn[0] = OaFnMatrix::Mamba3DtBwd(InDOut, x, DtMin_, DtMax_);
	}
private:
	OaF32 DtMin_;
	OaF32 DtMax_;
};

// ─── Empyrealm fused A·dt term (1:1 copy of Mamba3Adt, renamed) ──────────────
// Forward: ADT = min(-heavy_tail(ddA), -AFloor) * dt   (fused EmpyrealmAdt kernel).
// Both inputs are [B*S, H]; backward is pure elementwise (EmpyrealmAdtBwd kernel).
class OaGradEmpyrealmAdt final : public OaGradNode {
public:
	explicit OaGradEmpyrealmAdt(OaF32 InAFloor) noexcept : AFloor_(InAFloor) {}
	void Backward(const OaMatrix& InDOut, OaVec<OaMatrix>& OutDIn) override {
		const OaMatrix& ddA = Saved_[0];
		const OaMatrix& dt  = Saved_[1];
		auto g = OaFnMatrix::EmpyrealmAdtBwd(InDOut, ddA, dt, AFloor_);
		if (OutDIn.Size() > 0) OutDIn[0] = g.DDdA;
		if (OutDIn.Size() > 1) OutDIn[1] = g.DDt;
	}
private:
	OaF32 AFloor_;
};

// ─── Empyrealm fused dt term (1:1 copy of Mamba3Dt, renamed) ─────────────────
// Forward: DT = clamp(softplus(x), dt_min, dt_max)   (fused EmpyrealmDt kernel).
// Backward is pure elementwise (EmpyrealmDtBwd kernel).
class OaGradEmpyrealmDt final : public OaGradNode {
public:
	OaGradEmpyrealmDt(OaF32 InDtMin, OaF32 InDtMax) noexcept : DtMin_(InDtMin), DtMax_(InDtMax) {}
	void Backward(const OaMatrix& InDOut, OaVec<OaMatrix>& OutDIn) override {
		const OaMatrix& x = Saved_[0];
		if (OutDIn.Size() > 0) OutDIn[0] = OaFnMatrix::EmpyrealmDtBwd(InDOut, x, DtMin_, DtMax_);
	}
private:
	OaF32 DtMin_;
	OaF32 DtMax_;
};

// ─── Mamba-3 fused preprocess (split + rmsnorm + dt + adt in one kernel) ─────
// Forward: Mamba3Preprocess(projected, dtBias, config) → {X, Z, Bh, Ch, DT, ADT, Trap, Angle}
// Backward: Mamba3PreprocessBwd → {DProjected, DDtBias}
//
// Multi-output lazy-merge pattern: 8 thin grad nodes (one per output) share a
// SharedState. Each saves its output gradient and decrements a counter. The
// last one (counter→0) dispatches the fused backward and returns the result.
// The other 7 return empty matrices (skipped by the tape).
class OaGradMamba3Preprocess final : public OaGradNode {
public:
	struct SharedState {
		OaI32 Counter = 8;
		OaMatrix Projected, DtBias;
		OaMatrix DZ, DX, DBh, DCh, DDT, DADT, DTrap, DAngle;
		OaMatrix DProjected, DDtBias;
		OaFnMatrix::OaMamba3PreprocessConfig Config;
	};

	OaSharedPtr<SharedState> State_;
	OaI32 OutputIndex_ = 0;  // 0=z, 1=x, 2=bh, 3=ch, 4=dt, 5=adt, 6=trap, 7=angle

	void Backward(const OaMatrix& InDOut, OaVec<OaMatrix>& OutDIn) override {
		switch (OutputIndex_) {
			case 0: State_->DZ = InDOut; break;
			case 1: State_->DX = InDOut; break;
			case 2: State_->DBh = InDOut; break;
			case 3: State_->DCh = InDOut; break;
			case 4: State_->DDT = InDOut; break;
			case 5: State_->DADT = InDOut; break;
			case 6: State_->DTrap = InDOut; break;
			case 7: State_->DAngle = InDOut; break;
		}
		OaI32 remaining = --State_->Counter;
		if (remaining == 0) {
			auto grads = OaFnMatrix::Mamba3PreprocessBwd(
				State_->Projected, State_->DtBias,
				State_->DZ, State_->DX, State_->DBh, State_->DCh,
				State_->DDT, State_->DADT, State_->DTrap, State_->DAngle,
				State_->Config);
			State_->DProjected = grads.DProjected;
			State_->DDtBias = grads.DDtBias;
			if (OutDIn.Size() > 0) OutDIn[0] = grads.DProjected;
			if (OutDIn.Size() > 1) OutDIn[1] = grads.DDtBias;
		}
	}
};

// ─── Empyrealm fused preprocess (renamed copy of Mamba3Preprocess) ───────────

class OaGradEmpyrealmPreprocess final : public OaGradNode {
public:
	OaSharedPtr<OaGradMamba3Preprocess::SharedState> State_;
	OaI32 OutputIndex_ = 0;

	void Backward(const OaMatrix& InDOut, OaVec<OaMatrix>& OutDIn) override {
		switch (OutputIndex_) {
			case 0: State_->DZ = InDOut; break;
			case 1: State_->DX = InDOut; break;
			case 2: State_->DBh = InDOut; break;
			case 3: State_->DCh = InDOut; break;
			case 4: State_->DDT = InDOut; break;
			case 5: State_->DADT = InDOut; break;
			case 6: State_->DTrap = InDOut; break;
			case 7: State_->DAngle = InDOut; break;
		}
		OaI32 remaining = --State_->Counter;
		if (remaining == 0) {
			auto grads = OaFnMatrix::EmpyrealmPreprocessBwd(
				State_->Projected, State_->DtBias,
				State_->DZ, State_->DX, State_->DBh, State_->DCh,
				State_->DDT, State_->DADT, State_->DTrap, State_->DAngle,
				State_->Config);
			State_->DProjected = grads.DProjected;
			State_->DDtBias = grads.DDtBias;
			if (OutDIn.Size() > 0) OutDIn[0] = grads.DProjected;
			if (OutDIn.Size() > 1) OutDIn[1] = grads.DDtBias;
		}
	}
};

// ─── Mamba-3 SISO selective scan ─────────────────────────────────────────────

class OaGradMamba3Siso final : public OaGradNode {
public:
	OaFnMatrix::OaSsmConfig Config_{};
	void Backward(const OaMatrix& InDOut, OaVec<OaMatrix>& OutDIn) override {
		const OaMatrix& c      = Saved_[0];
		const OaMatrix& b      = Saved_[1];
		const OaMatrix& x      = Saved_[2];
		const OaMatrix& z      = Saved_[3];
		const OaMatrix& adt    = Saved_[4];
		const OaMatrix& dt     = Saved_[5];
		const OaMatrix& trap   = Saved_[6];
		const OaMatrix& angle  = Saved_[7];
		const OaMatrix& cbias  = Saved_[8];
		const OaMatrix& bbias  = Saved_[9];
		const OaMatrix& d      = Saved_[10];
		// Normalize dout shape: downstream views/reshapes (e.g. before out_proj MatMul,
		// or in the LM wrapper) may deliver a flat/3D/2D tensor sharing the siso output's
		// autograd meta. The Bwd host + kernel expect the original 4D output shape.
		OaMatrixShape expected = OaMatrixShape{Config_.Batch, Config_.SeqLen, Config_.NHeads, Config_.HeadDim};
		OaMatrix dOut4 = InDOut;
		if (InDOut.GetShape() != expected && InDOut.NumElements() == (OaI64)Config_.Batch * Config_.SeqLen * Config_.NHeads * Config_.HeadDim) {
			dOut4 = InDOut.Reshape(expected);
		}
		auto g = OaFnMatrix::Mamba3SisoBwd(dOut4, c, b, x, z, adt, dt, trap, angle,
			cbias, bbias, d, Config_);
		if (OutDIn.Size() > 0)  OutDIn[0]  = g.DC;
		if (OutDIn.Size() > 1)  OutDIn[1]  = g.DB;
		if (OutDIn.Size() > 2)  OutDIn[2]  = g.DX;
		if (OutDIn.Size() > 3)  OutDIn[3]  = g.DZ;
		if (OutDIn.Size() > 4)  OutDIn[4]  = g.DAdt;
		if (OutDIn.Size() > 5)  OutDIn[5]  = g.DDt;
		if (OutDIn.Size() > 6)  OutDIn[6]  = g.DTrap;
		if (OutDIn.Size() > 7)  OutDIn[7]  = g.DAngle;
		if (OutDIn.Size() > 8)  OutDIn[8]  = g.DCBias;
		if (OutDIn.Size() > 9)  OutDIn[9]  = g.DBBias;
		if (OutDIn.Size() > 10) OutDIn[10] = g.DD;
	}
};

// ─── Empyrealm post-gated SISO (Preprocess split path; correct gated bwd) ────
// The EmpyrealmSiso forward kernel folds the post gate (RmsNorm + silu(z) *
// norm_weight) into the scan output. The matching backward therefore must:
//   1. recompute the raw (ungated) scan output y,
//   2. reverse the post gate via RmsNormGatedBwd → dy_raw, dz_gate, d(norm_weight),
//   3. run the ungated Mamba3SisoBwd on dy_raw for the scan-input grads.
// (The old code reused OaGradMamba3Siso, which differentiates the UN-gated
//  output — wrong grads + zero norm_weight grad. This node fixes that.)
class OaGradEmpyrealmSiso final : public OaGradNode {
public:
	OaFnMatrix::OaSsmConfig Config_{};
	void Backward(const OaMatrix& InDOut, OaVec<OaMatrix>& OutDIn) override {
		const OaMatrix& c     = Saved_[0];
		const OaMatrix& b     = Saved_[1];
		const OaMatrix& x     = Saved_[2];
		const OaMatrix& z     = Saved_[3];
		const OaMatrix& adt   = Saved_[4];
		const OaMatrix& dt    = Saved_[5];
		const OaMatrix& trap  = Saved_[6];
		const OaMatrix& angle = Saved_[7];
		const OaMatrix& cbias = Saved_[8];
		const OaMatrix& bbias = Saved_[9];
		const OaMatrix& d     = Saved_[10];
		const OaMatrix& nw    = Saved_[11];

		const OaI32 B = static_cast<OaI32>(Config_.Batch);
		const OaI32 L = static_cast<OaI32>(Config_.SeqLen);
		const OaI32 H = static_cast<OaI32>(Config_.NHeads);
		const OaI32 P = static_cast<OaI32>(Config_.HeadDim);
		const OaI64 rows = static_cast<OaI64>(B) * L * H;

		// Recompute + reverse must not build graph (we are already inside backward;
		// the saved inputs are leaves that require grad, so unguarded forward ops
		// would attach spurious grad nodes and corrupt the scan-input gradients).
		OaGradNo noGrad;

		// Scan is ungated; the gate lives in the post block.
		OaFnMatrix::OaSsmConfig scanCfg = Config_;
		scanCfg.HasZ = 0u;

		// (1) Recompute raw scan output y (the fwd kernel only stored the gated y).
		auto y = OaFnMatrix::Mamba3Siso(c, b, x, z, adt, dt, trap, angle,
			cbias, bbias, d, scanCfg);                       // [B,L,H,P]
		auto yr = y.Reshape(OaMatrixShape{rows, P});
		auto zr = z.Reshape(OaMatrixShape{rows, P});
		auto ones   = OaFnMatrix::Ones(OaMatrixShape{P}, OaScalarType::Float32);
		auto noBias = OaFnMatrix::Zeros(OaMatrixShape{P}, OaScalarType::Float32);
		auto normGated  = OaFnMatrix::RmsNormGated(yr, ones, noBias, zr, 1e-5f, true);
		auto normGated4 = normGated.Reshape(OaMatrixShape{B, L, H, P});
		auto w4 = nw.Reshape(OaMatrixShape{1, 1, H, P});

		// (2) Reverse the post gate.
		auto dOut4 = InDOut.Reshape(OaMatrixShape{B, L, H, P});
		auto dNormGated2 = (dOut4 * w4).Reshape(OaMatrixShape{rows, P});
		auto gateGrad = OaFnMatrix::RmsNormGatedBwd(yr, ones, noBias, zr, dNormGated2, 1e-5f);
		auto dY4    = gateGrad.DX.Reshape(OaMatrixShape{B, L, H, P});
		auto dZGate = gateGrad.DZ.Reshape(z.GetShape());

		// (3) Ungated scan backward on the pre-gate gradient.
		auto sg = OaFnMatrix::Mamba3SisoBwd(dY4, c, b, x, z, adt, dt, trap, angle,
			cbias, bbias, d, scanCfg);

		auto dZ = (sg.DZ + dZGate).Reshape(z.GetShape());
		auto dNormWeight = OaFnMatrix::Sum(
			(dOut4 * normGated4).Reshape(OaMatrixShape{rows, P}), 0).Reshape(nw.GetShape());

		if (OutDIn.Size() > 0)  OutDIn[0]  = sg.DC;
		if (OutDIn.Size() > 1)  OutDIn[1]  = sg.DB;
		if (OutDIn.Size() > 2)  OutDIn[2]  = sg.DX;
		if (OutDIn.Size() > 3)  OutDIn[3]  = dZ;
		if (OutDIn.Size() > 4)  OutDIn[4]  = sg.DAdt;
		if (OutDIn.Size() > 5)  OutDIn[5]  = sg.DDt;
		if (OutDIn.Size() > 6)  OutDIn[6]  = sg.DTrap;
		if (OutDIn.Size() > 7)  OutDIn[7]  = sg.DAngle;
		if (OutDIn.Size() > 8)  OutDIn[8]  = sg.DCBias;
		if (OutDIn.Size() > 9)  OutDIn[9]  = sg.DBBias;
		if (OutDIn.Size() > 10) OutDIn[10] = sg.DD;
		if (OutDIn.Size() > 11) OutDIn[11] = dNormWeight;
	}
};

// OaGradEmpyrealmSisoFused REMOVED 2026-06-18 — fused experiment deleted.
// EmpyrealmSiso now uses OaGradMamba3Siso (identical math).
// ─── RoPE ───────────────────────────────────────────────────────────────────

class OaGradRoPE final : public OaGradNode {
public:
	OaI32 NumHeads_ = 0;
	OaI32 HeadDim_ = 0;
	OaF32 ThetaBase_ = 10000.0f;
	OaU32 PosOffset_ = 0;
	void Backward(const OaMatrix& InDOut, OaVec<OaMatrix>& OutDIn) override {
		if (OutDIn.Size() == 0) return;
		const OaMatrix& x = Saved_[0];
		OaI64 T = x.Size(0);
		OaU32 halfDim = static_cast<OaU32>(HeadDim_ / 2);
		OaU32 total = static_cast<OaU32>(T) * static_cast<OaU32>(NumHeads_) * halfDim;
		OaMatrix gradInput = OaFnMatrix::Empty(x.GetShape(), x.GetDtype());
		auto& ctx = OaContext::GetDefault();
		struct Push { OaU32 T, num_heads, head_dim; OaF32 theta_base; OaU32 pos_offset; }
			push{ static_cast<OaU32>(T), static_cast<OaU32>(NumHeads_), static_cast<OaU32>(HeadDim_), ThetaBase_, PosOffset_ };
		OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Write};
		ctx.Add("RopeApplyBwd", {&InDOut, &gradInput}, access, &push, sizeof(push), (total + 255u) / 256u);
		OutDIn[0] = gradInput;
	}
};
