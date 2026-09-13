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

The factor is `2^downsample_stages`. Fixed training windows divide it exactly;
whole-corpus tokenization accepts longer non-divisible clips and omits the
incomplete strided tail, matching the donor convolution geometry.
Channel normalization uses the donor's fused BCT operation directly. The
schema owns separate affine and affine-plus-ReLU forward/adjoint identities;
each forward replaces the former transpose–`LayerNorm`–transpose chain, while
each backward records one semantic operation over the fused row kernel and two
ordered affine reductions. The donor shader retains at most four channel
values per 256-lane invocation, so OARS rejects `C > 1024` instead of allowing
register-array overflow. Qualified timing remains separate from this
correctness result.

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
- donor-equivalent fused BCT channel normalization and fused ReLU against an
  independent host oracle, complete explicit and tape adjoints, the explicit
  1024-channel shader bound, and ALM encode without LayerNorm kernels;
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
- bounded host-side SafeTensors parsing and exact ViT-L/14 text-tower
  translation through `ClipText::import_safetensors`; unrelated vision weights
  are reported as unused while unknown text weights fail closed;
- frozen CLIP `OaClipTextAg` v1 `.oam` round-trip through canonical donor tensor
  paths and its exact packed 48-byte architecture payload;
- direct loading of the OA C++ 495 MB ViT-L/14 artifact, finite two-prompt
  Vulkan encoding, native product admission with its pinned merge asset, and a
  one-step tokenizer → uncached caption bake → conditioned-prior training gate;
- one product-level registered `tokenizer.*` / `prior.*` tree; and
- optimizer-free `OaAlmAg` v3 bundle save/load through the donor's exact packed
  205-byte architecture payload. Training-only EMA host counters remain in
  optimizer checkpoints rather than changing the C++ product state index.
- SDK-owned HumanML3D/CMP/KIT-ML host loading preserves little-endian C-order
  F32 NPY assets, Mean/Std standardization, all caption ranges, optional frozen
  text-feature identity/rows, and explicit clip upload. Independent host tests
  reproduce the donor's exact identity/perturbation gates for world-joint
  recovery, MPJPE, velocity error, contact accuracy, and foot skating.
- deterministic tokenizer windows preserve half-window overlap plus one exact
  tail window, while prior windows enumerate every valid start without
  manufacturing SOM/EOM at interior boundaries;
- hardware-backed tokenizer training executes reconstruction Smooth L1,
  temporal-velocity Smooth L1, rank-zero commitment loss, complete reverse
  mode, AdamW, cosine warmup, and one EMA transition per step;
- hardware-backed prior training executes masked true-boundary cross entropy,
  optional frozen caption features, MoE auxiliary loss/routing updates,
  complete reverse mode, AdamW, callbacks, and Vulkan timestamp evidence; and
- `train_alm` composes the two stages over the SDK HumanML3D owner, performs an
  explicit one-time token-corpus readback, validates cached text identity, and
  returns one product `Alm` ownership tree.
- `train_alm_with_native_text` validates the pinned ViT-L/14 tower and canonical
  merge table before either training stage, encodes every training and held-out
  caption in deterministic clip/caption order through frozen Vulkan CLIP
  batches of 16, and embeds that same tower plus merge bytes in the returned
  product. It does not require or silently consume a dataset feature cache.
- held-out tokenizer evaluation preserves the donor's non-wrapping final batch
  and reports reconstruction/velocity loss, MPJPE, contact accuracy, planted
  foot skating, live code count, and codebook perplexity;
- held-out prior evaluation preserves true sequence boundaries, uses the first
  cached caption row per clip, and reports valid-token-weighted cross entropy,
  perplexity, token accuracy, and EOM accuracy;
- `train_alm_with_validation` attaches both evaluators to the existing
  `Validation` callback lifecycle at epoch boundaries. Evaluation time is
  excluded from training throughput and the latest rich reports remain in the
  stage results for application output.
- both stage policies optionally attach the native `.oam` checkpoint callback.
  Resume restores the registered model tree, persistent tokenizer EMA buffers
  and scalar counter, complete AdamW moments/scalars, absolute iterator step,
  data-window cursor, scheduler position, and historical best-metric policy
  before admitting another update. `total_steps` remains the absolute final
  budget. The first step in a restarted process rebuilds transient allocation
  ordinals before stable-resource reuse.
- runnable Rust `ml_alm_train` and `ml_alm_generate` applications stage below
  `bin/<profile>/sdk/apps/ml/alm`. Training admits the donor `--val-split` and
  `--val-batches` policy and degrades explicitly to no validation when that
  split is unavailable. Conditioned training loads the donor-compatible native
  CLIP `.oam` and exact merges through `--clip-text-model` / `--clip-merges`;
  `--unconditional` is the explicit no-text route. Native checkpoint controls
  include directory, mid-epoch interval, retention, resume, restore-best, and
  explicit disable. The donor's VQ health values, AdamW peak/floor/warmup/decay
  policy, dense/MoE/hybrid architecture, routed-expert counts/cadence, and both
  MoE regularization policies are CLI-owned rather than hardcoded. Donor
  `--text-conditioning` and `--ckpt-*` spellings coexist with the shorter Rust
  compatibility flags and reach the same fields.
  Generation writes denormalized F32 NPY motion plus a transparent provenance
  sidecar; USD skeletal previews remain owned by the not-yet-ported USD/Render
  integration.

These are correctness and integration results, not a performance claim.

## Remaining parity

The following donor surfaces remain Planned:

- fused Conv1d/ReLU lowering and qualified channel-normalization timing;
- KV-cache generation;
- USD skeletal previews;
- donor checkpoint differential conversion, broader shapes/dtypes, validation
  layers, and canonical fresh-process performance qualification.

No remaining item is implied by the existence of the SDK facade.
