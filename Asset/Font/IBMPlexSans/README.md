# IBM Plex Sans

OA vendors `IBMPlexSans-Regular.ttf` as the source face for its persistent
SDF/MSDF UI and detection-label atlas.

- Upstream: <https://github.com/IBM/plex>
- Source copy: vendored from the Realm UI asset repository; upstream is IBM Plex.
- Version reported by the TTF: 3.005
- SHA-256:
  `975dcda37d80f038dcd143c22e33ca2d97a0cc5a929aace1c749153b0fe1afa5`
- License: SIL Open Font License 1.1; see `LICENSE.txt`

The TTF is a build-time input. Runtime code should consume the generated and
embedded atlas rather than loading this file from the filesystem.
