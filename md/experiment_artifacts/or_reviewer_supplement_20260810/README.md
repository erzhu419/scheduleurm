# Scheduleurm Lean Reviewer Supplement

This directory is the reviewer-facing Lean supplement for the Scheduleurm OR manuscript.
It is a flat upload wrapper around the checked consolidated proof file and build metadata.

## Files

- `ScheduleurmUpload.lean`: consolidated paper-facing Lean artifact.
- `lakefile.toml` or `lakefile.lean`: Lean project configuration copied from the proof root.
- `lean-toolchain`: exact Lean toolchain selector.
- `build.log`: captured output from the proof check command.
- `theorem_crosswalk.md`: manuscript theorem to Lean theorem-name crosswalk.
- `manifest.json`: machine-readable supplement metadata.
- `sha256sums.txt`: SHA-256 checksums for the supplement payload.

## Reproduction

Run `lake env lean ScheduleurmUpload.lean` from the proof root. The captured build status is `PASS` with return code `0`.

## Static Audit

`sorry`/`admit`/`axiom` static grep hits outside comments: `0`.
`ScheduleurmUpload.lean` SHA-256: `003b8551bb90b4532b454aa23e987940045c7d3197c13f4a1d2934efe3ecf9bc`.
