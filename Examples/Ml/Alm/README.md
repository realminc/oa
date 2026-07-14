# OaAlm

Two-stage discrete-token motion generation: VQ-VAE tokenizer + autoregressive LM.

See [dev.realm.software](https://dev.realm.software/) for the current architecture,
implementation status, and acceptance gates.

## Architecture

- **Stage 1: `OaAlmTokenizerAg`** — temporal-Conv1d VQ-VAE (8x downsample).
  Faithful to `mgpt_vq.py`. Reuses `OaResidualVectorQuantizer` (EMA codebook).
- **Stage 2: `OaAlmPriorAg`** — CLIP-conditioned autoregressive causal Transformer.
  Its FFN policy is dense by default and can use MoE or a hybrid cadence without
  replacing the ALM backbone or data pipeline.

## Files

NN modules live in `Extensions/` (compiled into `liboa`):

| File | Location | Purpose |
|---|---|---|
| `AlmConfig.h` | `Extensions/Public/Ml/Nn/Alm/` | Config structs for tokenizer, LM, dataset |
| `AlmTokenizerAg.h/.cpp` | `Extensions/{Public,Private}/Ml/Nn/Alm/` | Stage 1 VQ-VAE |
| `AlmPriorAg.h/.cpp` | `Extensions/{Public,Private}/Ml/Nn/Alm/` | Stage-2 Transformer LM with pluggable FFN policy |

This directory contains only the test suite:

| File | Purpose |
|---|---|
| `AlmAg.cpp` | GTest suite |
| `README.md` | This file |

## Build and run

```bash
cmake --build Build/Release --target AlmAg -j
./Bin/Release/Example/Ml/AlmAg
```

## Tests

- `Alm.ConfigTest` — config structs
- `Alm.TokenizerRoundTripShape` — 8x downsample shape + decode round-trip
- `Alm.ConvTranspose1dGradCheck` — bilinear identity grad check
- `Alm.Conv1dGradCheckStride1/2` — Conv1d grad checks
- `Alm.SingleConvIdentity` — single Conv1d learns identity
- `Alm.LinearAeSanity` — linear AE descent control
- `Alm.ComposedDescentCheck` — full encode/decode gradient descent direction
- `Alm.TokenizerLearnsRecon` — Gate 1: MSE drops and stays finite
- `Alm.LmStub` — LM constructs
- `Alm.GenerateStub` — generation pipeline stub
- `Alm.LmDynamicPrefixMatchesFullForward` — causal-mask and dynamic-sequence parity
- `Alm.LmFfnPoliciesForward` — dense, MoE, and hybrid FFN policy smoke test
- `Alm.LmCheckpointRoundtrip` — Transformer LM weights/optimizer persistence

## References

- [OA developer documentation](https://dev.realm.software/) — architecture and gates
- `Extensions/Public/Ml/Nn/Alm/AlmTokenizerAg.h` — tokenizer contract
- `Extensions/Public/Ml/Nn/Alm/AlmPriorAg.h` — conditional prior contract
- `<Oa/Ml/Nn.h>` — `OaResidualVectorQuantizer` (promoted to core)
