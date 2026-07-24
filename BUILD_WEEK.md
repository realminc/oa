# OA Mobile Lab — OpenAI Build Week 2026

**Track:** Developer Tools  
**Project:** [OA](https://github.com/realminc/oa)  
**Product surface:** OA Mobile Lab

OA Mobile Lab trains real neural networks locally on an Android phone through the same
C++ autograd engine, optimizer, checkpoint format, operation library, and Vulkan kernels
used by OA on desktop. It is a compact demonstration of a larger developer tool: one
cross-vendor execution substrate for building and validating GPU software without
maintaining a separate mobile ML implementation.

## What was built during Build Week

OA existed before the event. The baseline is private engineering commit `8e5a32b5`
from July 10, whose root `VERSION` was `0.7.2`; it was not published as a public
Git tag. The eligible Build Week work extends that foundation rather than
presenting five months of prior work as a one-week project. Sanitized public
event snapshots begin with `v0.7.3`.

| Area | Build Week extension |
|---|---|
| Android product | Built OA Mobile Lab: lifecycle-safe foreground training, live metrics, model selection, generation, cancellation, and checkpoint resume on a physical Adreno phone. |
| Shared validation | Unified Android and desktop around the same seeded corpus, dimensions, optimizer semantics, fixed prompt, generation gate, and `.oam` round trip. |
| Mobile Vulkan | Packaged a pinned, verified Turnip driver per app and added bounded mobile routes for recurrent and Mamba-3 backward execution. |
| Sparse execution | Reduced tiny sparse-MoE overhead from 46% to 8% on Byte and 46% to 12% on BPE while preserving dense-oracle gradients and end-to-end learning gates. |
| Training kernels | Added packed Transformer projections, Flash Attention, fused FFN/normalization paths, GPU-authored sparse workloads, and a cross-vendor descriptor-index correctness fix found by the release phone gate. |
| Runtime evidence | Added deterministic completed-execution graph reports with semantic operation provenance, selected implementation IDs, contract hashes, kernel-content hashes, resource lifetimes, alias groups, barriers, and completion timelines. |
| Engineering gates | Added enforceable module-dependency rules, release-safe Android report automation, real APK signing, and a sanitized public-snapshot pipeline that excludes internal engineering documents and credentials. |

Public `v0.7.3` (July 14) and `v0.7.4` (July 16) are dated intermediate snapshots. The
Build Week branch and release preserve the final eligible delta and its verification
evidence.

The July 21 `v0.7.6` preview publishes the completed private architecture-convergence
checkpoint as matching source and installable artifacts. Its controlled NLP comparison
predates the final release-source cleanup, so the results remain engineering evidence and
are not relabeled as an exact `v0.7.6` benchmark.

The July 24 `v0.7.7` preview is the post-event API convergence release: it adds
the canonical C++-parity Python root, paired tutorials, semantic media APIs, and
format-neutral still-image I/O. It does not alter or relabel the Build Week
measurements above.

## Demonstration

[Watch the public Build Week demo](https://www.youtube.com/watch?v=SEw20xx0SJY).

The canonical workload trains five architectures—RNN, GRU, Transformer, sparse-MoE
Transformer, and Mamba-3—for 300 AdamW updates with batch 64 and sequence length 16.
Every route must:

1. complete Vulkan forward, loss, backward, and optimizer execution;
2. reach the architecture's learning threshold;
3. generate 80 bytes autoregressively from the fixed prompt `to be`;
4. pass corpus-continuation and eight-gram coverage checks;
5. save an `.oam` checkpoint;
6. recreate the model and optimizer, reload the file, and reproduce generation exactly.

The gate intentionally rejects low teacher-forced loss accompanied by corrupt or
repetitive generation. It caught a thread-varying descriptor index in a packed QKV
kernel that desktop drivers tolerated and Turnip exposed.

The same repository also includes a small text-to-motion proof: text conditioning,
motion-token generation, decoding, and USD export through OaAlm. The current checkpoint
is an early model, not a production motion generator.

## Reproduce

Desktop reference:

```bash
cmake --preset release
ninja -C Build/Release TutorialNlpByteTransformerAg TutorialNlpByteMoeAg
OA_TUTORIAL_STEPS=300 Bin/Release/Tutorial/Ml/Nlp/TutorialNlpByteTransformerAg
OA_TUTORIAL_STEPS=300 Bin/Release/Tutorial/Ml/Nlp/TutorialNlpByteMoeAg
```

Android source build and five-model gate:

```bash
git submodule update --init --recursive Apps/Android/OaMobileLab/third_party/libadrenotools
cd Apps/Android/OaMobileLab
./gradlew assembleDebug
adb install -r ../../../Bin/Android/OaMobileLab/OaMobileLab-debug.apk
cd ../../..
Apps/Android/OaMobileLab/tools/run-nlp-suite.sh
```

For a signed release package, follow the environment-based signing contract in
[the Mobile Lab README](Apps/Android/OaMobileLab/README.md). The suite automatically
uses app-private report files for debug packages and bounded logcat records for the
non-debuggable release; both validate the same native report.

Detailed desktop protocol and results are in
[Docs/Benchmarks/OaNlpSuite.md](Docs/Benchmarks/OaNlpSuite.md).
The physical-device comparison and signed-release acceptance record are in
[Docs/Benchmarks/OaMobileLab.md](Docs/Benchmarks/OaMobileLab.md).

## Codex collaboration

OA is built by one developer working with Codex. During Build Week, Codex was used as an
engineering collaborator for architecture review, implementation, differential diagnosis,
test construction, documentation, and release automation. Hardware results were produced
by executing the code on the referenced Intel and Qualcomm devices; they are not generated
claims. Dated commits and reproducible commands define the human/AI collaboration boundary.

## Honest boundaries

- OA is a development preview; its C++, Python, and `.oam` compatibility contracts may
  change before 1.0.
- The Android reference is one Adreno 610 phone with app-local Turnip 26.1.4. It is a
  physical end-to-end gate, not a claim of universal Android compatibility.
- FP32 is the fully exercised path on the current Intel and Qualcomm hardware. BF16 and
  cooperative-matrix routes require capable devices and renewed vendor coverage.
- The educational NLP corpus demonstrates executor correctness and learning behavior,
  not production language-model quality.
- OA is Vulkan-first and cross-vendor; unsupported capabilities fail closed instead of
  silently selecting an unrelated CPU implementation.
