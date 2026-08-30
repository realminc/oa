# OA SDK

The SDK is OA's maintained source companion: examples, tutorials, reference
applications, sample-domain support, assets, and platform laboratories. The
installed framework remains under `source/`; SDK code consumes that framework
and does not define a second runtime.

```text
sdk/
  cpp/{applications,examples,tutorials,include,lib}
  py/{examples,tutorials,lib}
  android/mobilelab
  asset                       checked-in, manifest-verified SDK media
    docs/                    offline docs and publication images
  data/packs.toml             optional remote dataset descriptors
```

Production headers must not depend on this tree. Reusable product behavior is
promoted into `source/` only with an independent contract and tests; maintained
consumer examples remain here and feed generated documentation from checked code.

`sdk/asset` contains the small inputs required by tests, tutorials, generators,
and platform labs plus the media required by an offline documentation bundle
under `sdk/asset/docs`. Verify its complete inventory with:

```bash
python3 tools/data/checkAssets.py
```

Optional datasets are not Git or Git LFS payloads and are never downloaded by
library code. List and fetch a pinned pack explicitly:

```bash
python3 tools/data/manage.py list
python3 tools/data/manage.py fetch fashionMnist
```

Both C++ and Python resolve the resulting root through `oa::Paths::data` /
`oa.Paths.data`; set `OA_DATA_DIR` to move it outside the default `var/data`.
