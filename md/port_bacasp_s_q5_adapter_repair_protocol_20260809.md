# BACASP-S Q5 Adapter Repair Protocol

Date: 2026-08-09

## Observed Primary Holdout Result

The preregistered R89/R90 holdout remains a failed primary gate:

- registered instances: 54;
- completed instances: 36;
- parser failures: 18;
- every failure is a `Q=5` LargeMB instance whose source vessel table contains
  a Jumbo-vessel upper bound of six cranes;
- no holdout candidate search or policy retuning was performed.

The failed compact and full ledgers are retained under
`md/experiment_artifacts/port_bacasp_s_external_holdout_r89_r90_20260809.json`
and
`md/experiment_artifacts/port_bacasp_s_external_holdout_r89_r90_full_20260809.json.gz`.
They are not replaced by a repaired PASS.

## Source-Semantics Diagnosis

The BACASP-S paper defines `Q` as the number of available cranes and defines a
vessel-specific admissible interval from `q_i_min` through `q_i_max`. Its
LargeMB generator combines total-crane factors `Q in {5,10,15}` with the
unchanged Meisel--Bierwirth vessel classes, for which Jumbo vessels have
`q_i_max=6`. Consequently, the published Q5 files can contain a source upper
bound of six while only five specific cranes are physically available.

The current adapter rejects such a raw row during parsing. That is too strict:
the physically feasible action domain is the source vessel interval intersected
with the set of available cranes,

```text
{q_i_min, ..., q_i_max} intersect {1, ..., Q}.
```

The raw `u_iq` processing-time table must remain unchanged. The adapter may
enumerate only the intersection, and it must reject an instance if that
intersection is empty.

Primary source: J. F. Correcher et al., "The berth allocation and quay crane
assignment problem with crane travel and setup times," *Computers & Operations
Research* 162 (2024), 106468, DOI 10.1016/j.cor.2023.106468.

## Frozen Repair

Before any new holdout outcomes are inspected:

1. Preserve all raw vessel bounds and processing times in the parser.
2. Replace the invalid condition `q_i_max <= Q` with the nonempty-domain
   condition `q_i_min <= min(q_i_max,Q)`.
3. Expose the effective upper bound in the adapter audit and ensure every
   generated specific-crane action uses at most `Q` cranes.
4. Do not change the frozen trajectory
   `plan|spt_static+robust_maxweight+robust_maxweight`, baselines, metrics,
   bootstrap seed, or factor-cell denominator.
5. Generalize the evaluator only enough to bind a manifest's registered pair of
   replications instead of hard-coding R89/R90 labels.

## Untouched Confirmation Holdout

R87/R88, not R89/R90, will be the post-repair confirmation set. It will contain
the same complete 27-cell `Q x speed/setup x deadline` design and all 54 rows.
The raw files, manifest, implementation hashes, frozen plan, metrics, baselines,
failure policy, and bootstrap protocol must be preregistered after the repair is
implemented and before R87/R88 outcomes are evaluated.

R89/R90 may be rerun only as a labeled parser-repair diagnostic. It cannot be
reported as a second untouched holdout.

## Claim Boundary

A passing R87/R88 gate would support source-feasible transfer of the frozen
fixed-slot policy to a disjoint BACASP-S instance set. It would not establish
continuous-quay optimality, superiority to the authors' exact or genetic
algorithms, physical-port deployment, empirically calibrated migration costs,
or transfer of the server positive-recurrence theorem.
