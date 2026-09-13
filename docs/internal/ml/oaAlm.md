# OARS Animation Language Model

**Status:** Experimental connected SDK workload

**Updated:** 2026-09-13

**Donor authority:** OA C++ `sdk/cpp/include/ml/nn/alm`,
`sdk/cpp/lib/ml/nn/alm`, and `sdk/cpp/apps/gen3danim/trainalm.cpp`

## Ownership

ALM is a concrete product workload under `oa::sdk::ml::alm`; it is not a new
ML runtime or a generic NN namespace. `Alm` owns two required registered
children:

```text
Alm
├── tokenizer   temporal Conv1d VQ-VAE
├── prior       causal motion-token Transformer
├── text_encoder  optional frozen CLIP text tower
└── text_tokenizer_merges  optional persistent byte-BPE asset
```

An optional `text_encoder` child owns the frozen CLIP text-with-projection
tower when its projection width matches the conditioned prior. All children
borrow the caller's sole `Engine`. Their operations record into
the normal semantic/executable graph, their parameters and persistent buffers
use the ordinary `ModuleRegistry`, and a native `.oam` artifact traverses one
unambiguous `tokenizer.*` / `prior.*` tree. There is no ALM allocator,
scheduler, graph, or shader registry.

`from_external_text_parts` records an exact encoder identity while leaving
features caller-owned. `from_native_text_parts` admits only the pinned
`openai/clip-vit-large-patch14` tower and canonical merge table, then owns both
as one product. `encode_prompt` and `generate_motion_prompt` therefore need no
unrelated asset path or caller-assembled tokenizer.

## Tokenizer

`AlmTokenizer` preserves the donor topology:

```text
[B,T,input_dim]
  -> transpose [B,input_dim,T]
  -> Conv1d + channel LayerNorm + ReLU
  -> downsample_stages × {
       stride-2 Conv1d + channel LayerNorm + ReLU
       depth × dilated pre-norm residual block
     }
  -> Conv1d(code_dim)
  -> transpose and flatten [B*T/factor,code_dim]
  -> unit-RMS normalization
  -> one-level normalized EMA residual VQ
  -> symmetric residual Conv1d / ConvTranspose1d decoder
  -> [B,T,input_dim]
```

The factor is `2^downsample_stages`; input frame counts must divide it exactly.
Channel normalization is currently the differentiable
transpose–`LayerNorm`–transpose composition. This is mathematically equivalent
to the donor training fallback and intentionally remains visible until a fused
channel-normalization lowering has its own schema, adjoint, and performance
evidence.

Convolution weights consume one continuous donor-compatible LCG stream with
Glorot-uniform bounds. `new` uses the donor `0xC0FFEE` seed; `with_seed` makes
reproducible Rust tests and workloads explicit. Biases start at zero. The
latent RMS weight is a non-persistent buffer, while codebook, EMA sums,
populations, and the `u32` EMA transition count are persistent state.

Unlike the donor's empty compatibility `forward`, Rust `Module::forward`
performs the useful VQ-VAE reconstruction path. Explicit `encode`, `quantize`,
`decode`, `tokenize`, `detokenize`, `seed`, and `ema_update` retain the
stage-training boundaries.

## Prior

`AlmPrior` preserves the stage-two decoder topology and public policy:

- I32, U32, or U8 motion tokens enter a trainable token embedding;
- learned absolute positions are cached by `[batch,total_time]` geometry;
- an optional frozen-text feature is projected into one prefix token;
- every causal Transformer block uses a dense, MoE, or hybrid FFN according to
  `AlmFfnType` and one-based `moe_every` cadence;
- final RMSNorm feeds a bias-free vocabulary projection;
- conditioned logits exclude the text-prefix position;
- MoE auxiliary loss, routing-bias updates, and reduced route statistics are
  explicit methods rather than hidden training side effects.

Autoregressive generation begins every row with SOM, repeatedly evaluates the
growing causal prefix, samples the final logits, stops a row at EOM, and pads
completed rows. Nonpositive temperature selects deterministic greedy argmax;
positive temperature uses the existing Philox categorical path with an exact
per-step seed. `use_cache` is accepted as a reserved donor option, but no
KV-cache is claimed: both implementations currently replay the growing prefix.

`decode_to_motion` removes SOM, truncates every row to the shortest prefix
ending before EOM, and passes one dense code stream through the tokenizer. It
returns `None` when no motion-code position remains instead of manufacturing an
invalid empty Matrix.

## Evidence

Release-mode hardware gates on Intel Iris Xe, Mesa 26.2.2, Vulkan 1.4.354 prove:

- rank-two/rank-three tiled transpose and exact transpose adjoint;
- exact configured tokenizer shapes, unit-RMS latents, finite reconstruction,
  token lookup equivalence, and error rejection;
- backward reachability and finite gradients for every tokenizer parameter;
- exact `.oam` restoration of parameters, codebook buffers, and next EMA step;
- dense and hybrid-MoE prior forward, text-prefix shape behavior, complete
  parameter gradients, auxiliary loss, routing update, and route telemetry;
- deterministic greedy generation and SOM/code/EOM motion decoding;
- frozen CLIP token/position embedding, causal QuickGELU residual tower,
  explicit/fallback EOS selection, projection, and prior conditioning;
- native canonical 49,408-token CLIP byte-BPE parsing, BOS/EOS padding,
  truncation, and pinned OpenAI token-ID compatibility when the merge asset is
  supplied;
- frozen CLIP `OaClipTextAg` v1 `.oam` round-trip through canonical donor tensor
  paths and its exact packed 48-byte architecture payload;
- one product-level registered `tokenizer.*` / `prior.*` tree; and
- optimizer-free `OaAlmAg` v3 bundle save/load through the donor's exact packed
  205-byte architecture payload. Training-only EMA host counters remain in
  optimizer checkpoints rather than changing the C++ product state index.

These are correctness and integration results, not a performance claim.

## Remaining parity

The following donor surfaces remain Planned:

- fused channel-normalization and Conv1d/ReLU lowering with qualified timing;
- KV-cache generation;
- external CLIP weight translation into the native `.oam` model; native product
  bundles already own the merge table rather than requiring an application path;
- HumanML3D/KIT/CMP dataset, normalization, stage trainer, validation metrics,
  callbacks, and runnable `trainalm`/`genalm` applications;
- donor checkpoint differential conversion, broader shapes/dtypes, validation
  layers, and canonical fresh-process performance qualification.

No remaining item is implied by the existence of the SDK facade.
