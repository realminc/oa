# OA v0.17.0 — Experimental Rust and Python preview

## Summary

Breaking-rename and boundary release on top of v0.16.0. This release also
publishes the previously unreleased v0.16.1 milestone line: OpenCV-parity
vision milestones CV-1, CV-3, CV-6, and CV-7 — Canny edge detection with
connected-component labeling, distance transform, Hough transforms, and
external contours; ORB detect-describe with Hamming kNN matching;
`pyr_down`/`pyr_up`, Otsu/Triangle auto-threshold, and `integral`; CLAHE,
`calc_back_project`, HSV/Lab `convert_color` breadth, `moments`, `psnr`,
`ssim`, `match_template`, and `warp_polar`, all verified against host
oracles on Iris Xe.

## Compatibility and migration

- `oa::Engine` is renamed `oa::Runtime` across the public API; Python
  `oa.Engine` becomes `oa.Runtime`. `current`, `set_default`, and the
  `*_on` method family keep their semantics.
- `oa::matrix::mat_mul*` and `oa.matrix.mat_mul*` are renamed to the
  `matmul` family. The weight-layout `[M,K] x [N,K] -> [M,N]` GEMM
  formerly spelled `mat_mul_nt` is now `matrix::matmul` /
  `Matrix::matmul` / `oa.matrix.matmul`.
- The `oa.ml.nlp` and `oa.ml.audio` Python modules and their bindings
  are removed. The corresponding model and dataset workloads live in the
  standalone oasdk source companion as plain consumers of `oa.ml.nn`.
- The `sdk/` source tree is removed from this repository; runnable
  tutorials, benchmarks, and examples ship with the SDK repository.
- `oa::Path::asset` additionally probes `src/asset/` and `asset/` for
  the relocated SDK asset layout.

## Verification

Release-profile build and the `cargo test --release` suite pass on the
release commit (0 failures; hardware-gated entries remain ignored).
`cargo fmt --check`, `cargo clippy --all-targets --all-features -- -D
warnings`, and `git diff --check` are clean on the tagged commit. All 16
Python NLP tutorials and the audio-commands tutorial run end-to-end
against the shipped surface on Intel Iris Xe (ADL GT2, Vulkan), with 15
of 16 reproducing their Rust oracle constants exactly.

## Known limitations

Compiler tile providers and hardware video are Experimental. Multi-device
workloads are usable through `set_default`/`_on` but distributed execution
is not yet supported. The `oasdk` companion repository is not yet
published; this release removes its source from the `oa` tree.

# OA v0.16.0 — Experimental Rust and Python preview

## Summary

Minor release for v0.15.3. Completes the PyTorch `nn` module surface:
`oa::ml::nn` now exports 95 modules covering convolution (including
3-D and transposed), the full pooling and unpooling family, padding
modes, dropout variants, normalization, recurrent layers, attention
and transformers, embeddings, shape utilities, and composite models.
Loss coverage grows to fifteen `oa::ml::loss` functions including the
CTC forward-backward dynamic-programming kernels and a composed
`AdaptiveLogSoftmaxWithLoss` module.

Additional milestones shipped in this line: fused LSTM kernels with
one-dispatch BPTT, incremental Transformer decoding with an attention
KV cache, encoder-decoder cross-attention, grouped-query and
multi-query attention variants, parameter-level gradient clipping,
device-resident AMP gradient scaling, ONNX graph import with named
state round-trip, `QuantizedLinear` with a BF16 capability query,
async-prefetch dataset iteration, image histogram statistics, and new
ViT and residual ConvNet tutorials.

## Compatibility and migration

- `oa::ml::matrix` transposed-convolution functions are renamed to the
  `conv_<dim>_transpose` family (`conv_1d_transpose`, `conv_2d_transpose`,
  `conv_3d_transpose`), matching the `conv_1d_bias_*` fusion naming.
  The `nn::ConvTranspose1d/2d/3d` module names are unchanged.
- PipeWire audio capture moves behind an optional Cargo feature; builds
  without the feature compile a stub path (this also unblocks docs.rs).
- All other changes are additive.

## Verification

Release-profile build and the `cargo test --release` suite pass on the
release commit (`0` failures; hardware-gated entries remain ignored).
On Intel Iris Xe (ADL GT2, Vulkan), the new module and loss surface is
exercised by device tests: transposed-convolution adjoints, pool/unpool
oracles, padding backward preimage oracles, CTC against a brute-force
alignment enumeration plus finite-difference backward, and
`AdaptiveLogSoftmaxWithLoss` against a two-level nll oracle with
parameter-gradient checks. `cargo fmt --check`,
`cargo clippy --all-targets --all-features -- -D warnings`, the
operation/nn generator suites (259 tests, 1174 subtests), and
`git diff --check` are clean on the tagged commit.

## Known limitations

Compiler tile providers and hardware video are Experimental. Multi-engine
workloads are usable through `set_default`/`_on` but distributed execution
is not shipped. Sampling utilities (top-k/top-p/beam search), the
diffusion primitive set, ONNX export, and the model zoo remain planned.
Additional device qualification and precision-specific training gates
remain open.

## Upgrade

Use matching Rust, Python and SDK versions. Verify the published artifact
checksums before installing from GitHub Releases.
