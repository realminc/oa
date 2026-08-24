# Intel One Mono

OA vendors the Regular and Medium faces. Regular is the source face for the
hinted monospace UI-text atlas; both weights are staged into the Android Mobile
Lab from this one canonical directory.

- Upstream: <https://github.com/intel/intel-one-mono>
- Source copy: the existing Realm Editor font asset, byte-identical to the
  audited OA input recorded below.
- Version reported by the TTF: 1.4
- SHA-256:
  `131ab6a8f6e8b9160bc526353262828bc24caab8fcdcfd7a9edc7a044e974230`
- License: SIL Open Font License 1.1; see `OFL.txt`

The TTF is a build-time input. Runtime code consumes the generated embedded
coverage atlas and does not open this file.
