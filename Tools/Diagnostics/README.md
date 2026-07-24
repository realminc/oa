# OA Diagnostic Evidence

`oaevidence.py` creates one machine-readable, checksummed directory containing
the evidence needed to reproduce an OA Vulkan result. It uses only the Python
standard library and does not add a runtime dependency.

Snapshot the current checkout and Vulkan device:

```bash
python3 Tools/Diagnostics/oaevidence.py
```

Wrap a workload, request validation, and have OA export the first completed
training graph automatically:

```bash
python3 Tools/Diagnostics/oaevidence.py \
  --validation \
  --output var/report/transformer-validation \
  -- Bin/Release/Tutorial/Ml/Nlp/TutorialNlpTransformer
```

Run the focused execution-graph suite under a named validation profile and
require the exact feature set to be observed with zero validation errors:

```bash
python3 Tools/Diagnostics/run_validation.py --mode sync
python3 Tools/Diagnostics/run_validation.py --mode gpu
```

The available profiles are `core`, `sync`, `gpu`, and `all`. GPU-assisted
validation instruments shaders and is intentionally opt-in. A profile run uses
the existing Release binary by default: `OA_VK_VALIDATION=1` enables the layer
at runtime, while `OA_VK_VALIDATION_MODE` selects its diagnostic features.
The evidence manifest records the selected validation-layer API and
implementation version plus any explicit `VK_LAYER_PATH`/`VK_ADD_LAYER_PATH`.

GPU-assisted shared-memory race validation requires a layer containing
Khronos VVL commit `4cd431278be1e3dd074af9989e956306e4f1a2a6` (first upstream
version tag `v1.4.354`). Earlier layers can report a known false race between a
single invocation's own shared-memory load/store; see [VVL issue
#12415](https://github.com/KhronosGroup/Vulkan-ValidationLayers/issues/12415).
The runner never suppresses the diagnostic or converts the run to a pass. It
recognizes only the exact affected same-invocation signature, leaves the gate
red, and prints the required tool upgrade.

Attach existing graph, validation, and benchmark artifacts:

```bash
python3 Tools/Diagnostics/oaevidence.py \
  --graph var/report/training_graph.json \
  --validation-log var/report/validation.log \
  --benchmark var/report/transformer.json \
  --capture-reference var/capture/transformer.gfxr
```

Collect stable GEMM selection records and fail if an unapproved precision,
layout, or naive fallback occurs:

```bash
python3 Tools/Diagnostics/oaevidence.py \
  --selection-trace \
  --output var/report/transformer-selection \
  -- Bin/Release/Tutorial/Ml/Nlp/TutorialNlpTransformer
```

Intentional universal-layout or tiny-naive paths can be declared explicitly
with repeatable `--allow-fallback layout` / `--allow-fallback naive` options.
The manifest contains aggregate kernel and fallback counters; normal workloads
pay no diagnostic logging or counter cost.

The manifest records:

- Git commit, branch, dirty state, OA version, and selected CMake configuration;
- host platform and a deliberately small allowlist of relevant OA/Vulkan
  environment variables;
- the discovered `vk.xml` path, size, and SHA-256;
- `vulkaninfo --summary` and the Vulkan Profiles JSON for the actual device;
- workload command, duration, exit status, logs, validation request/observation,
  and generated execution graph;
- checksummed graph, validation, and benchmark inputs;
- optional capture references without copying multi-gigabyte captures.

The collector never overwrites an existing directory. It builds in a temporary
sibling and renames the completed bundle atomically. Arguments whose option
name contains `token`, `secret`, `password`, or `api-key` are redacted from the
manifest. A validation request is not reported as enabled unless OA's runtime
log confirms that the validation layer was active.

Run its tests with:

```bash
python3 Tools/Diagnostics/test_oaevidence.py
```

## Rested benchmarks

`oabench.py` is the canonical fresh-process performance protocol. Short
workloads require at least seven measured runs. The runner preserves warm-up
and measured logs; records Git, build, Vulkan, CPU-governor and power-profile
metadata; and emits median, median absolute deviation, p10/p90 and total spread.

```bash
python3 Tools/Diagnostics/oabench.py \
  --output var/report/compute-graph-process.json \
  --name runtime.compute_graph_process \
  --contract tests=25 \
  --warmup 2 --runs 7 --cooldown 2 \
  --require-regex '\[  PASSED  \] 25 tests' \
  -- Bin/Release/Test/Runtime/Graph/TestComputeGraph
```

Use `--metric-regex`, `--metric-name`, and `--metric-unit` to extract the last
matching workload metric instead of process wall time. A regex must have one
capture group or a named `value` group. Supplying `--baseline` compares matching
workloads and fails only when a regression exceeds both the default 3% policy
and the larger observed relative-MAD noise band.

Comparisons fail closed unless the logical command, complete workload contract,
selected Vulkan device/driver family, relevant build options, metric, and unit
are identical. A faster number from another GPU or a differently compiled
binary is evidence, but it is not a valid comparison.

The checked-in suite names the six accepted vertical workloads and resolves a
baseline directory from the selected hardware identity:

```bash
python3 Tools/Diagnostics/oabenchsuite.py --list
python3 Tools/Diagnostics/oabenchsuite.py \
  --output-dir /tmp/oa-benchmark-candidate
```

The suite covers canonical FP32 GEMM, a complete dense Transformer block,
sparse-MoE backward, upload-ring and persistent-readback paths, and AV1 decode
with its PSNR oracle still enabled. It runs each workload in fresh processes;
the inner benchmark is unchanged and the outer runner supplies two settling
runs, seven measured runs, and a two-second cooldown.

Baseline acceptance is an explicit maintainer action and is permitted only for
a passing result captured from a clean repository with at least seven samples
and a complete Vulkan platform identity:

```bash
python3 Tools/Diagnostics/oabenchsuite.py \
  --accept \
  --accept-reason 'initial Iris Xe release baseline' \
  --output-dir /tmp/oa-benchmark-accept
```

`oabaseline.py` writes compact `oa.benchmark_baseline.v1` records containing
the source commit and result hash. It never overwrites an accepted baseline.
A deliberate replacement therefore remains reviewable: remove the old record,
capture from a clean checkpoint, state the reason, and commit the new artifact.
Checked-in records normalize the maintainer's home-directory prefix to `~` in
`VCPKG_INSTALLED_DIR`; the measured values, build profile, source commit, and
result hash remain unchanged. This removes personal workstation identity
without inventing a different build root.

## Determinism

`oadeterminism.py` runs a fixed-seed workload in fresh processes and compares
only explicitly selected output traces. It preserves checksummed logs and
records the Git revision, build configuration, host and relevant environment.
Volatile timestamps and unrelated diagnostics therefore cannot create false
failures.

```bash
python3 Tools/Diagnostics/oadeterminism.py \
  --output var/report/rnn-determinism.json \
  --name ml.rnn_copy_loss \
  --trace-regex 'step [0-9]+ loss=(?P<value>[0-9.]+)' \
  --trace-regex 'copy-task: (?P<value>.*)' \
  --require-regex '\[  PASSED  \] 1 test' \
  -- Bin/Release/Test/Ml/Nn/TestRnn \
     --gtest_filter=Rnn.RnnTrainsOnSequenceTask
```

Every trace regex must match at least once, and its complete ordered match list
must equal the first run. This is an exact evidence gate, not a tolerance-based
performance comparison.

## Host sanitizers

ASAN and UBSAN use independent CMake, binary, output, and vcpkg installed trees.
They exercise ownership, containers, threading, validation, dtype, model-weight
parsing, tokenization, and configuration parsing without requiring a GPU:

```bash
cmake --preset asan
cmake --build Build/Asan --target \
  TestTypes TestMemory TestContainers TestThreading TestCoreMisc TestValidation \
  TestPrecisionDtype TestTransferWeights TestOamIntegrity TestTokenizer TestGptOssConfig TestOaStd
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1:strict_string_checks=1 \
  ctest --test-dir Build/Asan --output-on-failure --timeout 120 -L core

cmake --preset ubsan
cmake --build Build/Ubsan --target \
  TestTypes TestMemory TestContainers TestThreading TestCoreMisc TestValidation \
  TestPrecisionDtype TestTransferWeights TestOamIntegrity TestTokenizer TestGptOssConfig TestOaStd
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  ctest --test-dir Build/Ubsan --output-on-failure --timeout 120 -L core
```

The sanitizer profiles instrument OA and their test executables. Vulkan
validation and GPU-assisted validation remain separate gates for device-side
access and synchronization errors.

## Device admission canary

`oa-device-canary` runs a short deterministic FP32/UInt32 known-answer workload
before a device is admitted to a valuable job. The standalone build-tree form
writes only `oa.device_canary.v1` JSON to stdout; OA diagnostics remain on
stderr, so the report can be redirected directly:

```bash
Bin/Release/Tutorial/Core/TutorialCoreDeviceCanary \
  > var/report/device-canary.json
python3 -m json.tool var/report/device-canary.json
```

The installed-package command is `oa-device-canary`. Exit status `0` means all
sampled checks passed, `3` means a known-answer mismatch (`DataLoss`), and `2`
means the canary could not establish evidence because engine, allocation,
dispatch, synchronization, or readback failed.

Run the same canary under the two Vulkan correctness profiles with:

```bash
python3 Tools/Diagnostics/run_validation.py --mode sync -- \
  Bin/Release/Tutorial/Core/TutorialCoreDeviceCanary
python3 Tools/Diagnostics/run_validation.py --mode gpu -- \
  Bin/Release/Tutorial/Core/TutorialCoreDeviceCanary
```

This canary samples representative transport, synchronization, reduction, and
GEMM paths. It does not prove arbitrary arithmetic correct and does not replace
checkpoint hashes, periodic testing, guarded rollback, or independent shadow
execution.
