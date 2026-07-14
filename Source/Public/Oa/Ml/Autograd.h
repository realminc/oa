// OaGradAuto — Reverse-mode AD umbrella header.
//
// Include this header to get all autograd types:
//   - OaGradNode / OaFnAutograd / OaGradNo / OaGradientTape (from Core)
//   - All OaGrad* concrete nodes (from Matrix / Loss / Optim)
//
#pragma once

// Single include pulls in Core + Matrix + Loss + Optim autograd headers
#include "../../../Private/Oa/Ml/Autograd/Autograd.gen.h"
