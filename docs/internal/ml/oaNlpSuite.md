# OARS NLP tutorial suite

**Status:** Experimental Char-RNN and Char-Transformer rows

**Updated:** 2026-09-08

## Implemented Char-RNN row

The first Rust row is the canonical character Elman RNN:

```text
U32 characters [64, 16]
  -> Embedding(27, 32)
  -> Rnn(32, 64, one layer)
  -> Linear(64, 27) at all 1,024 positions
  -> mean cross-entropy
  -> complete BPTT
  -> AdamW(0.01)
```

It has exactly 8,891 trainable scalars under seven registration-derived paths.
The sampler uses the exact OA C++ 576-character, three-repeat corpus, cursor
advance, batch-row stride of seven, context length 16, batch 64, and shifted
all-position targets. The tutorial performs exactly 300 optimizer steps and
greedily generates 80 characters from prompt `to be` with the same
left-filled/sliding-window policy as the C++ Char tutorial.

The current unqualified local Iris Xe Release validation produced:

```text
loss: initial 3.310754 final 0.185840
accuracy: 92.5%
generated: "to be that is the question whether tis nobler in the mind to suffer the slings and ar"
wall: 63.39 ms/step
GPU: mean 62.298 ms/step · p50 61.592 · p95 67.912 ms
```

The final loss, accuracy, and generated continuation match the C++ reference
quality behavior. The wall result does not: the C++ reference measured about
10.25 ms/step under its fixed-clock benchmark protocol. OARS currently uses a
correct deterministic RNN backward baseline in which one workgroup serializes
the complete batch. This single run is correctness and callback-integration
evidence, not a qualified performance comparison.

## Acceptance and reproduction

The ignored Vulkan integration test fixes the corpus, dimensions, parameter
paths/count, first batch/target values, 300-step count, final loss below `0.25`,
accuracy above 90%, and exact greedy continuation.

```sh
cargo test --release --test ml nlp::canonical_char_rnn_completes_the_cpp_300_step_gate \
  -- --ignored --test-threads=1 --nocapture

cargo build --release --example ml_nlp_char_rnn
python3 tools/build/stage.py --profile release --target ml_nlp_char_rnn
./bin/release/sdk/tutorials/ml/nlp/ml_nlp_char_rnn
```

## Implemented Char-Transformer row

The Rust model matches the C++ tutorial topology and registration tree:

```text
token Embedding(27, 32) + position Embedding(16, 32)
  -> TransformerBlock(32, 64, sequence 16, one causal head)
     -> pre-LayerNorm -> Q/K/V projections -> causal SDPA -> output projection -> residual
     -> pre-LayerNorm -> Linear(32,64) -> GELU -> Linear(64,32) -> residual
  -> final LayerNorm -> Linear(32,27)
```

It owns the same 22 parameter tensors and 10,875 trainable scalars. GELU,
residual addition, LayerNorm, and causal Q/K/V attention all participate in the
complete reverse path. Independent tests cover GELU and two-head causal
attention forward values and finite-difference gradients. The acceptance gate
captures the fixed-shape forward/backward/AdamW step once and refreshes stable
input slots without reauthoring the semantic graph. Untimed replay uses the
reusable command cache. The metric-enabled tutorial uses a fresh
timestamp-wrapped command per replay because one query pair cannot be reset
while an earlier asynchronous submission may still own it. It
also saves all registration-addressed parameters plus AdamW moments/configuration,
loads them into fresh owners, and reproduces optimizer step, accuracy, and text.

One serial Release comparison on the same Iris Xe produced:

| implementation | initial loss | final evaluation loss | accuracy | wall |
| --- | ---: | ---: | ---: | ---: |
| OA C++ | 5.059494 | 0.190238 | 92.7% | 9.82 ms/step |
| OARS eager baseline | 4.040805 | 0.193185 | 92.4% | 158.067 ms/step |
| OARS donor-tiled kernels + timed captured replay | 4.040804 | 0.193185 | 92.4% | 8.91 ms/step |

Both generated exactly:

```text
"to be or not to be that is the question whether tis nobler in the mind to suffer the "
```

The timed OARS run also reported GPU mean 7.369 ms/step, p50 3.441 ms, p95
24.597 ms, and a 17% wall-to-GPU gap. The differing C++/Rust initial loss
reflects different deterministic initializer streams; it is not stepwise
numerical identity. The current OARS wall result is one exploratory run under
uncontrolled thermal state, not the fixed-clock seven-process protocol, so it
is not yet a qualified performance comparison with the C++ row. Final quality
and greedy behavior match exactly.

```sh
cargo test --release --test ml nlp::canonical_char_transformer_completes_the_cpp_300_step_gate \
  -- --ignored --exact --test-threads=1 --nocapture

cargo build --release --example ml_nlp_char_transformer
python3 tools/build/stage.py --profile release --target ml_nlp_char_transformer
./bin/release/sdk/tutorials/ml/nlp/ml_nlp_char_transformer
```

The shared `ItTraining` lifecycle and progress/summary callbacks are now used by
both Char tutorials. The GRU primitive is implemented separately, but no GRU
tutorial row is admitted here yet. Fixed-clock fresh-process performance
qualification and the remaining Byte/BPE × RNN/GRU/Transformer/MoE/Mamba-3
rows remain Planned.
