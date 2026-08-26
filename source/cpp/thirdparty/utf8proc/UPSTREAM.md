# utf8proc provenance

OA vendors the utf8proc 2.11.3 release as a private Unicode implementation
dependency for text shaping and the SDK CLIP tokenizer.

- Upstream: <https://github.com/JuliaStrings/utf8proc>
- Version: `2.11.3`
- Tag: `v2.11.3`
- Archive SHA-512: `148701fce506d076f03497b6d085f1993eff743debad4a2f6d3cbac91e19a5c22d9938245bdb460c1b22b51842c7416c42124db7416c684ee63d622490baac0e`
- License: MIT; see `LICENSE`

| File | SHA-256 |
|---|---|
| `utf8proc.c` | `edf80118fd34796bcabef4567cc684adebc7058f2afe8cfdc11b7a78e383ed84` |
| `utf8proc.h` | `a4e498b7392c383cf3b22e662da21e0b48f1264806235d87b5d8bd166232f658` |
| `utf8proc_data.c` | `950e549dbfc853c4304425f3af1875e72fa9fc9697c273c763400c2da4e380a7` |
| `LICENSE` | `3b510150d34f248a221bb88e1d811238d6c6c18b51231822c42974c39bb07256` |

`utf8proc.c`, `utf8proc.h`, `utf8proc_data.c`, and `LICENSE` are an unmodified
extraction of that release. OA compiles the C source directly into its runtime;
do not patch this snapshot without a documented upstream defect, conformance
test, and an explicit delta recorded here.
