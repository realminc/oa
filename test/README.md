# Test layout

Tests are grouped by language, then owning module, matching OA's language split.
Rust uses snake_case `test_*.rs`; generated tests retain the `.gen.rs` suffix.

```text
test/
  rs/
    test_core.rs          Cargo suite: core
    test_runtime.rs       Cargo suite: runtime
    test_matrix.rs        Cargo suite: matrix
    core/test_memory.rs   module-local test names
    runtime/test_vk.rs
    matrix/test_blas.gen.rs
  py/
    test_inventory.py
    tools/gen/fn/test_generate.py
    tools/profiling/test_bench.py
```

The three Rust suite files explicitly register their module files using `#[path]`.
Cargo's automatic integration-test discovery is disabled. The Python inventory
gate checks that every Rust test source is registered exactly once. Do not add an
unregistered file or depend on Cargo recursively discovering these directories.
Private Rust unit tests remain next to the implementation under `#[cfg(test)]`;
they are compiled only by the library test harness, not by release library builds.

```bash
cargo test --all-features
cargo test --test core memory
cargo test --test runtime
cargo test --test matrix blas:: -- --ignored --nocapture
python3 -m unittest discover -s test/py -v
python3 tools/gen/fn/generate.py --check
```

Hardware tests keep their explicit capability/validation requirements and ignore
annotations; reorganizing the suite does not turn an unrun GPU test into a pass.
Python currently tests repository tooling. OA Python-binding tests belong under
`test/py/<module>/` when those bindings exist; no placeholder binding suite is
claimed here. Add `__init__.py` to nested Python test directories for discovery.
