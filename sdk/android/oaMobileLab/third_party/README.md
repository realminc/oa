# Android probe third-party components

## libadrenotools

- Source: <https://github.com/bylaws/libadrenotools>
- Commit: `8fae8ce254dfc1344527e05301e43f37dea2df80`
- Nested `liblinkernsbypass`: `aa3975893d83ef1bc84c321ec60c65fbf1287887`
- License: BSD-2-Clause (`libadrenotools/LICENSE`)

The source is tracked as a recursive Git submodule. OA uses only the custom-driver loader
and its required hook libraries.

## Mesa Turnip Android driver

- Build source: <https://github.com/nihui/mesa-turnip-android-driver>
- Release: `26.1.4`, published 2026-07-03
- Artifact: `mesa-turnip-android-26.1.4.zip`
- Archive SHA-256: `a559b4257d7964e8082d1bfcfc3fb77ea95cd1f8071ee6e0b3e6c0d859539fa0`
- Driver SHA-256: `5a9bdaa51e31c4579dfea7217039bca65d99ff9aeee503f55860939039c4f043`

`tools/fetch-turnip.sh` downloads and verifies the artifact. The extracted binary is a
generated, ignored build input and is not stored in OA Git history.

Turnip is an app-local experimental driver for this probe. It does not replace the
device's system driver and is not yet an OA production runtime dependency.
