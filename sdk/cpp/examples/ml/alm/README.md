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

NN modules live in `extensions/` (compiled into `liboa`):

| File | Location | Purpose |
|---|---|---|
| `almConfig.h` | `sdk/cpp/include/ml/nn/alm/` | Config structs for tokenizer, LM, dataset |
| `almTokenizerAg.h/.cpp` | `extensions/{include,lib}/ml/nn/alm/` | Stage 1 VQ-VAE |
| `almPriorAg.h/.cpp` | `extensions/{include,lib}/ml/nn/alm/` | Stage-2 Transformer LM with pluggable FFN policy |

This directory contains only the test suite:

| File | Purpose |
|---|---|
| `AlmAg.cpp` | GTest suite |
| `README.md` | This file |

## Build and run

```bash
cmake --build build/release --target AlmAg -j
./bin/release/example/ml/almAg
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
- `sdk/cpp/include/ml/nn/alm/almTokenizerAg.h` — tokenizer contract
- `sdk/cpp/include/ml/nn/alm/almPriorAg.h` — conditional prior contract
- `<Oa/Ml/Nn.h>` — `OaResidualVectorQuantizer` (promoted to core)
