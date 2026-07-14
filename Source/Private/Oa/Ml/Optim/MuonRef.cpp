// OaMuonRef — CPU reference implementation for Muon.

#include <Oa/Ml/MuonRef.h>
#include <Oa/Ml/Module.h>

#include <cmath>
#include <cstring>
#include <vector>

namespace OaMuonRef {

namespace {

constexpr OaF32 kNsA = 3.4445f;
constexpr OaF32 kNsB = -4.7750f;
constexpr OaF32 kNsC = 2.0315f;

OaF32 FrobeniusNorm(const float* InData, OaU32 InCount, OaF32 InEps) {
	double sumsq = 0.0;
	for (OaU32 i = 0; i < InCount; ++i) {
		const double v = static_cast<double>(InData[i]);
		sumsq += v * v;
	}
	return std::sqrt(static_cast<OaF32>(sumsq) + InEps);
}

void Transpose(
	float* Out,
	const float* In,
	OaU32 InRows,
	OaU32 InCols) {
	for (OaU32 r = 0; r < InRows; ++r) {
		for (OaU32 col = 0; col < InCols; ++col) {
			Out[static_cast<size_t>(col) * InRows + r] = In[static_cast<size_t>(r) * InCols + col];
		}
	}
}

} // namespace

namespace {

bool SegmentEquals(OaStringView InSegment, OaStringView InName) {
	return InSegment == InName;
}

bool IsAuxAdamSegment(OaStringView InSegment) {
	static constexpr const char* kAux[] = {
		"embed", "wte", "token_embed", "byte_embed",
		"head", "lm_head", "output", "classifier",
		"bias",
	};
	for (const char* name : kAux) {
		if (SegmentEquals(InSegment, name)) return true;
	}
	return false;
}

} // namespace

bool IsMuonMatrixParam(OaStringView InPath, const OaParameter& InParam) {
	if (!InParam.RequiresGrad) return false;
	if (InParam.Data.Rank() != 2) return false;

	const OaString path(InPath);
	OaUsize start = 0;
	while (start < path.size()) {
		const OaUsize dot = path.find('.', start);
		const OaUsize len = (dot == OaString::npos) ? (path.size() - start) : (dot - start);
		const OaStringView segment(path.c_str() + start, len);
		if (IsAuxAdamSegment(segment)) return false;
		if (dot == OaString::npos) break;
		start = dot + 1;
	}
	return true;
}

OaOfficialMuonSplit SplitOfficialRouting(OaSpan<const OaNamedParameter> InNamed) {
	OaOfficialMuonSplit split;
	for (const OaNamedParameter& named : InNamed) {
		if (!named.Param || !named.Param->RequiresGrad) continue;
		if (IsMuonMatrixParam(named.Path, *named.Param)) {
			split.Muon.PushBack(named.Param);
		} else {
			split.AdamW.PushBack(named.Param);
		}
	}
	return split;
}

OaF32 MoonshotScale(OaU32 InRows, OaU32 InCols, OaF32 InRmsMatch) {
	const OaU32 maxDim = (InRows > InCols) ? InRows : InCols;
	return InRmsMatch * std::sqrt(static_cast<OaF32>(maxDim));
}

void NewtonSchulz5(
	float* OutOrtho,
	const float* InUpdate,
	OaU32 InRows,
	OaU32 InCols,
	OaI32 InNS5Steps,
	OaF32 InEps) {
	const OaU32 count = InRows * InCols;
	const bool transposed = InRows > InCols;
	const OaU32 operRows = transposed ? InCols : InRows;
	const OaU32 operCols = transposed ? InRows : InCols;

	std::vector<float> z(static_cast<size_t>(operRows) * operCols);
	if (transposed) {
		Transpose(z.data(), InUpdate, InRows, InCols);
	} else {
		std::memcpy(z.data(), InUpdate, static_cast<size_t>(count) * sizeof(float));
	}

	const OaF32 norm = FrobeniusNorm(z.data(), operRows * operCols, InEps);
	for (float& v : z) {
		v /= norm;
	}

	auto zidx = [operCols](OaU32 r, OaU32 c) -> size_t {
		return static_cast<size_t>(r) * operCols + c;
	};

	for (OaI32 step = 0; step < InNS5Steps; ++step) {
		std::vector<float> a(static_cast<size_t>(operRows) * operRows, 0.0f);
		for (OaU32 i = 0; i < operRows; ++i) {
			for (OaU32 k = 0; k < operCols; ++k) {
				const float zik = z[zidx(i, k)];
				for (OaU32 j = 0; j < operRows; ++j) {
					a[static_cast<size_t>(i) * operRows + j] += zik * z[zidx(j, k)];
				}
			}
		}

		std::vector<float> aa(static_cast<size_t>(operRows) * operRows, 0.0f);
		for (OaU32 i = 0; i < operRows; ++i) {
			for (OaU32 k = 0; k < operRows; ++k) {
				const float aik = a[static_cast<size_t>(i) * operRows + k];
				for (OaU32 j = 0; j < operRows; ++j) {
					aa[static_cast<size_t>(i) * operRows + j] += aik * a[static_cast<size_t>(k) * operRows + j];
				}
			}
		}

		std::vector<float> b(static_cast<size_t>(operRows) * operRows, 0.0f);
		for (OaU32 i = 0; i < operRows * operRows; ++i) {
			b[i] = kNsB * a[i] + kNsC * aa[i];
		}

		std::vector<float> newZ(static_cast<size_t>(operRows) * operCols, 0.0f);
		for (OaU32 i = 0; i < operRows; ++i) {
			for (OaU32 k = 0; k < operCols; ++k) {
				newZ[zidx(i, k)] = kNsA * z[zidx(i, k)];
			}
		}
		for (OaU32 i = 0; i < operRows; ++i) {
			for (OaU32 k = 0; k < operRows; ++k) {
				const float bik = b[static_cast<size_t>(i) * operRows + k];
				for (OaU32 j = 0; j < operCols; ++j) {
					newZ[zidx(i, j)] += bik * z[zidx(k, j)];
				}
			}
		}
		z.swap(newZ);
	}

	if (transposed) {
		Transpose(OutOrtho, z.data(), operRows, operCols);
	} else {
		std::memcpy(OutOrtho, z.data(), static_cast<size_t>(count) * sizeof(float));
	}
}

void MatrixStep(
	float* InOutWeights,
	float* InOutMomentum,
	const float* InGrads,
	OaU32 InRows,
	OaU32 InCols,
	OaF32 InLr,
	OaF32 InBeta,
	OaF32 InWeightDecay,
	OaF32 InEps,
	OaI32 InNS5Steps,
	OaF32 InRmsMatch) {
	const OaU32 count = InRows * InCols;

	std::vector<float> nesterov(count);
	for (OaU32 i = 0; i < count; ++i) {
		const float g = InGrads[i];
		const float mNew = InBeta * InOutMomentum[i] + (1.0f - InBeta) * g;
		nesterov[i] = (1.0f - InBeta) * g + InBeta * mNew;
		InOutMomentum[i] = mNew;
	}

	std::vector<float> ortho(count);
	NewtonSchulz5(ortho.data(), nesterov.data(), InRows, InCols, InNS5Steps, InEps);

	const OaF32 scale = MoonshotScale(InRows, InCols, InRmsMatch);
	for (OaU32 i = 0; i < count; ++i) {
		InOutWeights[i] = (1.0f - InLr * InWeightDecay) * InOutWeights[i]
			- InLr * scale * ortho[i];
	}
}

void VectorStep(
	float* InOutWeights,
	float* InOutMomentum,
	const float* InGrads,
	OaU32 InCount,
	OaF32 InLr,
	OaF32 InBeta,
	OaF32 InWeightDecay) {
	for (OaU32 i = 0; i < InCount; ++i) {
		const float g = InGrads[i];
		const float mNew = InBeta * InOutMomentum[i] + (1.0f - InBeta) * g;
		const float update = (1.0f - InBeta) * g + InBeta * mNew;
		InOutMomentum[i] = mNew;
		InOutWeights[i] = (1.0f - InLr * InWeightDecay) * InOutWeights[i] - InLr * update;
	}
}

} // namespace OaMuonRef