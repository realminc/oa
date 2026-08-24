# OA SDK

The SDK is OA's maintained source companion: examples, tutorials, reference
applications, sample-domain support, assets, and platform laboratories. The
installed framework remains under `source/`; SDK code consumes that framework
and does not define a second runtime.

```text
sdk/
  cpp/{applications,examples,tutorials,include,lib}
  py/{examples,tutorials,lib}
  android/oaMobileLab
  asset
```

Production headers must not depend on this tree. Reusable product behavior is
promoted into `source/` only with an independent contract and tests; maintained
consumer examples remain here and feed generated documentation from checked code.
