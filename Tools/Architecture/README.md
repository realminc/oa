# OA Architecture Checks

`oacheck.py` makes OA's source-layer dependency direction enforceable without depending
on the CMake build. The checker and its policy files are the public executable contract.

```bash
python3 Tools/Architecture/oacheck.py
python3 Tools/Architecture/test_oacheck.py
```

`modules.json` is the target dependency graph. `dependency_baseline.json`
contains the maximum accepted count for violations that already existed at the
`v0.6.98` recovery baseline. A count may decrease, but increasing it or adding a
new forbidden edge fails CI.

When an existing violation is removed, lower or delete its baseline entry in
the same commit. Never raise a baseline to make a new dependency pass; change
the dependency or record an architecture decision.
