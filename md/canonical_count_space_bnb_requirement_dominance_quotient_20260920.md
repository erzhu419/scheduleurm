# Canonical count-space B&B requirement-dominance quotient

Implemented an independent wrapper around the existing reducer and score-exact
tie continuation. The fixed nine-option hardware catalogue prepares to eight
options with retained original indices `(0,2,3,4,5,6,7,8)` and dominance
witness `1 -> 0`. Preparation is queue-independent and reusable across frames.

Each prepared selection maps primary and final actions back to the original
nine-option catalogue. Count vectors are expanded to nine positions with
option 1 fixed at zero, while LP calls, matrix coefficients, candidate
evaluations, represented configurations, and represented count vectors remain
the actual eight-option B&B work. Reduction and certificate-lift preprocessing
have separate probes, coordinate-read, and comparison counters.

The full-catalogue score certificate follows from equal requirements and a
strictly shorter effective duration. Canonical lifting additionally requires a
strictly smaller original index for every retained replacement; preparation
rejects catalogues that lack this proof. The per-frame result exposes both the
static lift proof and the lifted primary/final exactness flags.

Validation: 18 focused tests pass. They cover one-time preparation reuse,
ordinary and triggered continuation paths, original-index reconstruction,
eight-option work identity, full K4096 action/score/certificate parity on four
registered states including the empty queue, and invalid catalogue, queue,
budget, and idle-duration inputs. No runner, artifact, existing reducer/core,
or paper file was changed.
