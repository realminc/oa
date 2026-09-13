# OARS NLP tutorial suite

**Status:** Experimental canonical 5×3 comparison suite complete

**Updated:** 2026-09-12

## Canonical suite matrix

The Rust SDK now implements the complete donor comparison matrix: RNN, GRU,
Transformer, sparse-MoE Transformer, and Mamba-3 over Char, raw Byte, and BPE
tokenization. All 15 rows run the same 576-byte corpus, `[64, 16]` batches,
300 optimizer steps, all-position cross-entropy, fixed-prompt greedy generation,
and native checkpoint reload. BPE rows additionally round-trip their
`oa_bpe_v1` vocabulary.

| Tokenization | RNN | GRU | Transformer | sparse MoE | Mamba-3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Char | complete | complete | complete | complete | complete |
| Byte | complete | complete | complete | complete | complete |
| BPE | complete | complete | complete | complete | complete |

“Complete” here means the deterministic single-device correctness gate and
runnable tutorial are connected. It is not a fixed-clock performance-parity or
cross-device qualification claim.

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

## Implemented Char-MoE Transformer row

The Rust MoE row ports the donor's exact character recipe:

```text
token Embedding(27, 32) + position Embedding(16, 32)
  -> pre-norm one-head causal attention + residual
  -> RMSNorm -> router(32, 4) -> deterministic top-2 routing
  -> four stacked SwiGLU experts with DFF=16 -> residual
  -> final LayerNorm -> Linear(32, 27)
```

It owns 23 parameter tensors and 13,183 scalars, matching OA C++. The accepted
Release hardware gate captures forward, complete backward, and AdamW once;
replays the graph for 300 steps with 598 stable input uploads; evaluates loss
and accuracy; generates 80 characters from `to be`; and reloads parameters,
persistent routing state, and optimizer moments from `.oam`.

The first unqualified local Iris Xe tutorial run produced initial loss
`3.514067`, final training loss `0.190367`, final evaluation loss `0.183673`,
92.578% accuracy, 5.89 ms wall per step, and 4.940 ms mean GPU time:

```text
"to be that is the question whether tis nobler in the mind to suffer the slings and ar"
```

That is within the OA C++/Android reference regime of loss `0.1907` and 93.2%
accuracy. It was not run under the fixed-clock seven-process protocol, so the
timings are an absolute observation rather than a qualified OA C++ comparison.

```sh
cargo test --release --test ml \
  nlp::canonical_char_moe_transformer_completes_the_cpp_300_step_gate \
  -- --ignored --exact --test-threads=1 --nocapture

cargo build --release --example ml_nlp_char_moe_transformer
python3 tools/build/stage.py --profile release --target ml_nlp_char_moe_transformer
./bin/release/sdk/tutorials/ml/nlp/ml_nlp_char_moe_transformer
```

## Implemented Char-Mamba-3 row

The Rust row preserves the donor character model's 10,915 trainable scalars:

```text
Embedding(27, 32)
  -> Mamba3(model 32, inner 64, state 32, heads 4, groups 1, effective rank 1)
  -> gated RMSNorm inside the Mamba block
  -> Linear(32, 27)
```

It trains for exactly 300 AdamW steps at learning rate 0.003 through the same
`ItTraining` metrics/progress/summary lifecycle, evaluates the complete fixed
corpus, greedily generates 80 characters from `to be`, and verifies a native
`.oam` checkpoint in fresh owners. The deterministic Intel Iris Xe gate reports
initial loss 3.303064, final training loss 0.194512, evaluation loss 0.190578,
92.9688% accuracy, and exact continuation:

```text
"to be that is the question whether tis nobler in the mind to suffer the slings and ar"
```

One unqualified Release tutorial run completed in 2.46 seconds, or 8.19 ms per
step wall time, with 7.482 ms mean GPU time. This is a functional diagnostic,
not a fixed-clock multi-process performance comparison.

```sh
cargo test --release --test ml \
  nlp::canonical_char_mamba3_completes_the_cpp_300_step_gate \
  -- --ignored --exact --test-threads=1 --nocapture

cargo build --release --example ml_nlp_char_mamba3
python3 tools/build/stage.py --profile release --target ml_nlp_char_mamba3
./bin/release/sdk/tutorials/ml/nlp/ml_nlp_char_mamba3
```

The shared `ItTraining` lifecycle and progress/summary callbacks are used by all
15 canonical NLP tutorials plus the Empyrealm-Core fidelity row. Deterministic
`oa_bpe_v1` tokenization, raw U8 upload, packed-U8 Embedding forward/adjoint,
ByteEmbedding/ByteHead, and exact byte-logit decoding are connected as documented in
[Byte and BPE NLP](oaByteNlp.md). Every Char, Byte, and BPE architecture now
completes the same 300-step, evaluation, generation, and checkpoint contract.
The fidelity row preserves its distinct module/parameter topology but shares
the Mamba-3 operation providers because the donor shader bodies are identical.
Fixed-clock fresh-process performance qualification and cross-device coverage
remain Planned.

## Implemented Byte and BPE rows

The raw-byte rows preserve the donor's 256-token vocabulary and canonical
all-position sampler. They accept packed U8 inputs while retaining U32
cross-entropy targets. BPE learns 64 deterministic merges, compresses the
576-byte corpus to 210 tokens, records exact source-byte throughput, and stores
its vocabulary separately from model/optimizer ownership. Every row captures
the complete forward/backward/AdamW program, reuses one command recording for
299 subsequent untimed replays, generates exact bytes, and reloads fresh owners.

```text
Byte RNN: loss 5.569197 -> 0.186080 · accuracy 92.1875%
Byte GRU: loss 5.550348 -> 0.499343 · accuracy 85.5469%
Byte Transformer: loss 5.656755 -> 0.190565 · accuracy 92.8711%
Byte MoE: loss 5.686717 -> 0.193275 · accuracy 92.5781%
Byte Mamba-3: loss 5.539418 -> 0.207903 · accuracy 92.9688%
BPE RNN: loss 5.774095 -> 0.020861 · accuracy 98.8281%
BPE GRU: loss 5.766479 -> 0.021224 · accuracy 98.8281%
BPE Transformer: loss 5.871271 -> 0.020028 · accuracy 98.9258%
BPE MoE: loss 5.844332 -> 0.019962 · accuracy 98.8281%
BPE Mamba-3: loss 5.774934 -> 0.019415 · accuracy 98.9258%
```

```sh
cargo test --release --test ml nlp::canonical_byte_rnn_completes_the_cpp_300_step_gate \
  -- --ignored --exact --test-threads=1 --nocapture
cargo test --release --test ml nlp::canonical_byte_gru_completes_the_cpp_300_step_gate \
  -- --ignored --exact --test-threads=1 --nocapture

cargo build --release --examples
python3 tools/build/stage.py --profile release
```
