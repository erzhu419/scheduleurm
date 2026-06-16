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
- `no_sorry_audit.txt`: static source audit for `sorry`, `admit`, and `axiom`.
- `sha256sums.txt`: SHA-256 checksums for the supplement payload.

## Reproduction

Run `lake env lean ScheduleurmUpload.lean` from the proof root. The captured build status is `PASS` with return code `0`.

## Static Audit

`sorry`/`admit`/`axiom` static grep hits outside comments: `0`.
The explicit audit command and result are recorded in `no_sorry_audit.txt`.
`ScheduleurmUpload.lean` SHA-256: `25f0c744fb10fe8a9c3399f5072d5d16a81084082ab0264e92e9e7c8d08739c2`.
