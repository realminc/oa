#pragma once

#include <Oa/Ml/Module.h>
#include <Oa/Ml/Optim.h>

/// Hyperparameters for official Muon + AdamW dual-optimizer training.
struct OaMuonAdamWConfig {
	OaF32 MuonLr = 2e-2f;
	OaF32 AdamWLr = 3e-4f;
	OaF32 MuonBeta = 0.95f;
	OaF32 MuonWeightDecay = 0.1f;
	OaF32 MuonEps = 1e-7f;
	OaI32 MuonNs5Iters = 5;
	OaF32 AdamWBeta1 = 0.9f;
	OaF32 AdamWBeta2 = 0.95f;
	OaF32 AdamWEps = 1e-8f;
	OaF32 AdamWWeightDecay = 0.01f;
};

/// Split named params and build OaOptimizerComposite (Muon on 2D body, AdamW on aux).
[[nodiscard]] OaUniquePtr<OaOptimizerComposite> MakeMuonAdamWOptimizer(
	OaModule& InModel,
	const OaMuonAdamWConfig& InCfg = {});