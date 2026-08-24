# xsimd provenance

OA vendors the header-only xsimd 14.1.0 release as a private implementation
dependency for `oa::FnSimd`.

- Upstream: <https://github.com/xtensor-stack/xsimd>
- Version: `14.1.0`
- Archive: `xtensor-stack-xsimd-14.1.0.tar.gz`
- SHA-512: `a7030787e49bfb8fd544c94abf5e44ebdf4d922298e7ecbe9a51c473a956f3461d877611b4b0e2f5e962815f513589a124045e3909c864d0ba0d86a96a46ad21`
- License: BSD-3-Clause; see `LICENSE`

The `include/xsimd` tree and `LICENSE` are an unmodified extraction of that
release. OA-specific policy and operations remain in
`source/cpp/lib/oa/core/simd.cpp`; do not patch the vendored headers without a
documented defect, platform test, benchmark, and an explicit delta recorded in
this file.
