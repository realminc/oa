# Release and publication

**Status:** Canonical release-lane and artifact contract

OA uses two deliberately independent semantic-version lanes while the C++ to
Rust port is incomplete:

| Repository | Lane | Meaning |
| --- | --- | --- |
| `empyrealm/oars` | `0.1.x`, then `0.2.x` | private Rust engineering milestones |
| `realminc/oa` | `0.8.x` | sanitized public previews and published packages |

The versions are expected to differ. They converge only through an explicit
future release decision after the port, not through ad-hoc renaming. Published
public tags and package versions are immutable; a failed public release advances
to the next patch.

## Public snapshot

Create a public candidate from a committed private milestone without mutating
the checkout:

```bash
tools/release/createPublicSnapshot.sh v0.1.8 0.8.2 public/main
```

The command prints a new commit ID. It archives the exact private commit,
removes private agent/editor configuration, translates only the canonical
Rust, Python, lockfile, and release-document version sources, rejects symlinks,
credential-shaped paths, secret-shaped content, and personal workstation paths,
and parents the result to the nominated public commit. The marker file in the
result records both versions and the private source commit.

Before updating `public/main`, inspect the emitted commit and prove that its
tree builds and tests. Tag that commit with the public version (`v0.8.2` in the
example), push only the named public branch and tag, and wait for the release
workflow to reach a terminal successful state.

## Artifact contract

A tagged public workflow currently produces:

- an exact source archive and resolved Cargo lockfile;
- a Linux x86-64 archive of runnable Rust SDK tutorials, benchmarks, and apps;
- `oa-sdk` Debian, RPM, and Arch packages containing those programs;
- a CPython 3.10+ ABI3 `oapython` wheel containing the native extension;
- dependency and build-toolchain evidence; and
- one checksum manifest over every attached artifact.

Rust has no stable language ABI, and OA does not yet expose a supported C ABI.
The root crate is therefore source-linked into Rust consumers; it is not shipped
as a misleading internal-ABI `liboa.so`. A distinct `oa` runtime package is
Planned only after a real dynamic-library boundary, headers or equivalent ABI
metadata, install/import tests, and compatibility policy exist. This is the
intentional Rust replacement for the old C++ split runtime/SDK packages, not a
claim that the old runtime artifact has already been ported.

The wheel is independently installed into a clean environment before PyPI
publication. CI imports every packaged domain, checks root/module type identity,
compiles the Python surface, and audits the archive inventory without claiming
that a software Vulkan runner qualifies hardware execution. Physical-device
Python tests remain a separate release gate. The release job then downloads
that exact PyPI wheel, checks its digest, attaches it beside the native packages
and evidence, regenerates the complete checksum manifest, and anonymously
downloads and verifies every published asset.
