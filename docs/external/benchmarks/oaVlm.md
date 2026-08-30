# OA Vulkan Linear Math benchmark

**Status:** Qualified single-host development evidence

**Updated:** 2026-08-30

This report compares the admitted packed CPU `oa::vlm` spatial-math surface
with GLM 1.1.0. It is a workload-specific engineering result, not a universal
or fastest-library claim. GLM is an external benchmark input and is not an OA
product or package dependency.

## Correctness and comparison boundary

The benchmark covers 50 operation families in both `F32` and `F64`. Each case
executes its complete OA and GLM workloads before timing, rejects non-finite or
mismatched checksums, alternates OA-first and GLM-first sample order, and
rejects unknown command-line arguments.

Cases are classified as:

- **arithmetic** — equivalent valid-input work suitable for direct performance
  comparison;
- **hardened** — OA's checked/fail-closed contract against a raw GLM operation.
  These expose validation cost but are not equivalent-security regressions.

OA retains packed scalar values and does not expose xsimd types or alignment.
The compiler may vectorize visible fixed-shape formulas under the qualified
native Release build. Explicit batch-array SIMD remains separately owned by
`oa::FnSimd`.

## Provenance and method

The GLM source archive reports version 1.1.0. Its sorted `glm/` file-hash
manifest has SHA-256
`2134e2062c2a98ee535a16308780dccfd844f5945d303369a83547a0c8109c49`.

| Field | Qualified value |
|---|---|
| OA source | Private arithmetic checkpoint `v0.7.9`; release repairs in sanitized public `v0.7.29` |
| Compiler/build | Clang 22.1.8; C++20; `-O3 -DNDEBUG -march=native` |
| Host | Intel Core i5-1145G7; 4 physical cores / 8 threads; Linux 7.1.8 |
| Pinning/power | CPU 2; `intel_pstate` powersave; balanced platform profile |
| Workload | 65,536 deterministic values; 50 cases; `F32` and `F64` |
| Inner estimator | 3 warmups; 15 samples; median; alternating order |
| Outer estimator | 2 process warmups; 7 measured fresh processes; 3-second cooldown |
| Thermal gate | package at most 65 C before each process; measured starts 58-64 C |
| Correctness | all 100 precision/case oracles passed |
| Stability metric | `F32 Mat4` multiply: 8.714 ns/item median; 1.83% relative MAD; 13.34% spread |

The ratio is OA time divided by GLM time; lower is better. Parity is at most
1.03x. Across the 27 equivalent arithmetic cases, all 27 reached parity or
lower time in both precisions. The geometric-mean ratio was 0.866x for `F32`
and 0.881x for `F64`, corresponding to 13.4% and 11.9% lower time.

## Equivalent arithmetic results

| Operation | F32 OA / GLM ns | F32 ratio | F64 OA / GLM ns | F64 ratio |
|---|---:|---:|---:|---:|
| `vec2_add` | 1.003 / 1.003 | 1.000x | 1.011 / 1.014 | 0.997x |
| `vec2_dot` | 1.003 / 1.003 | 1.000x | 0.986 / 1.022 | 0.965x |
| `vec3_add` | 0.935 / 0.996 | 0.938x | 1.962 / 1.990 | 0.986x |
| `vec3_component_mul` | 0.973 / 1.004 | 0.969x | 1.804 / 1.900 | 0.950x |
| `vec3_dot` | 1.005 / 1.004 | 1.001x | 1.604 / 1.658 | 0.967x |
| `vec3_cross` | 1.076 / 1.130 | 0.952x | 1.815 / 1.836 | 0.988x |
| `vec3_reflect` | 1.871 / 2.148 | 0.871x | 2.481 / 2.617 | 0.948x |
| `vec3_lerp` | 1.134 / 1.380 | 0.822x | 1.942 / 2.091 | 0.929x |
| `vec3_refract` | 3.084 / 3.086 | 0.999x | 3.363 / 3.510 | 0.958x |
| `vec4_add` | 1.147 / 1.124 | 1.020x | 3.145 / 3.183 | 0.988x |
| `vec4_dot` | 1.038 / 1.091 | 0.951x | 2.660 / 2.679 | 0.993x |
| `quat_mul` | 2.572 / 2.522 | 1.020x | 3.470 / 3.782 | 0.918x |
| `mat3_vec3` | 2.406 / 2.604 | 0.924x | 4.936 / 5.196 | 0.950x |
| `mat3_add` | 3.813 / 3.858 | 0.988x | 6.909 / 6.973 | 0.991x |
| `mat3_transpose` | 1.927 / 1.884 | 1.023x | 4.398 / 4.441 | 0.990x |
| `mat3_mul` | 5.392 / 5.753 | 0.937x | 7.754 / 8.715 | 0.890x |
| `mat3_determinant` | 1.611 / 1.595 | 1.010x | 4.464 / 4.418 | 1.010x |
| `mat4_vec4` | 4.820 / 5.043 | 0.956x | 8.153 / 8.447 | 0.965x |
| `mat4_add` | 7.595 / 7.678 | 0.989x | 11.921 / 12.105 | 0.985x |
| `mat4_transpose` | 4.928 / 5.106 | 0.965x | 8.543 / 8.541 | 1.000x |
| `mat4_mul` | 8.714 / 14.015 | 0.622x | 13.530 / 13.880 | 0.975x |
| `mat4_determinant` | 5.830 / 5.679 | 1.026x | 9.499 / 9.294 | 1.022x |
| `transform_point` | 4.777 / 4.892 | 0.977x | 8.353 / 8.578 | 0.974x |
| `transform_direction` | 4.726 / 4.875 | 0.969x | 8.323 / 8.778 | 0.948x |
| `compose_trs` | 8.263 / 23.068 | 0.358x | 11.437 / 35.469 | 0.322x |
| `translation_matrix` | 0.912 / 2.183 | 0.418x | 0.940 / 1.992 | 0.472x |
| `scale_matrix` | 0.960 / 2.120 | 0.453x | 1.031 / 2.282 | 0.452x |

The largest reductions come from explicitly unrolled fixed-shape work, direct
affine TRS construction, and translation/scale constructors that avoid generic
matrix transforms. Ordinary finite normalization and determinant paths retain
out-of-line robust fallbacks for exceptional magnitudes and non-finite input.

## Hardened-operation characterization

OA's checked operations normalize non-unit quaternions, support finite
subnormal and extreme magnitudes, reject invalid tolerances and degenerate
geometry, estimate inverse conditioning, verify general inverse residuals,
preserve outputs on failure, and fail closed from direct APIs. Raw GLM peers do
not perform equivalent work.

| Operation family | F32 ratio range | F64 ratio range | Interpretation |
|---|---:|---:|---|
| Vector length/distance/normalize | 0.981-1.791x | 0.981-1.252x | Scale-safe exceptional-value handling and checks |
| Checked vector project/angle | 1.162-1.558x | 1.083-1.388x | Tolerance, degeneracy, finiteness and output preservation |
| Quaternion checked/integration | 1.185-2.792x | 1.216-2.524x | Unit validation, shortest path and fail-closed normalization |
| Checked matrix inverse/decomposition | 1.105-4.880x | 1.198-4.503x | Conditioning, finiteness, residual and shear/reflection policy |
| Camera/projection/viewport | 1.339-17.404x | 1.479-2.742x | Input validation and checked per-point inverse |

Per-point viewport unprojection is the clearest remaining optimization seam:
OA currently performs a checked general inverse for every point. The safe next
step is a checked precomputed-inverse or bulk-unproject contract, not removal of
singularity, residual, finiteness, or output-preservation guarantees.

## Reproduce

```bash
cmake -S . -B build/release \
	-DOA_VLM_GLM_ROOT=/absolute/path/to/glm-1.1.0
cmake --build build/release --target BenchVlm TestVlm
python3 tools/diagnostics/oaBench.py \
	--output var/benchmark/vlm-glm.json \
	--name core.vlm_glm_suite \
	--command-id benchVlm_glm_1_1_0_f32_f64 \
	--contract peer=glm-1.1.0 --contract precision=f32-f64 \
	--contract cases=50 --contract items=65536 \
	--contract oracle=whole-case-checksum --contract order=alternating \
	--warmup 2 --runs 7 --cooldown 3 --timeout 300 \
	--thermal-limit-celsius 65 --thermal-sensor-regex 'Package id 0' \
	--thermal-timeout 300 \
	--metric-regex 'PAIR precision=f32 case=mat4_mul contract=arithmetic oa_ns=([0-9.]+)' \
	--metric-name mat4_mul_oa --metric-unit ns/op \
	--require-regex 'BENCHMARK oracle=PASS' \
	--cmake-cache build/release/CMakeCache.txt -- \
	taskset -c 2 bin/release/test/core/vlm/benchVlm \
	--items 65536 --warmups 3 --samples 15 --precision both
```
