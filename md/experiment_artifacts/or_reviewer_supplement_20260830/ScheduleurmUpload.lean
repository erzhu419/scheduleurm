import Mathlib
import Mathlib.Topology.MetricSpace.HausdorffDistance

/-!
# Scheduleurm consolidated Lean proof file

Generated mechanically from `Scheduleurm/*.lean` for upload.
The original split files are unchanged. This file intentionally imports only mathlib,
not local `Scheduleurm.*` modules.
-/



/-! ## Source: Scheduleurm/Basic.lean -/


/-!
# Scheduleurm: Basic finite configuration model

This file sets up the finite-dimensional objects used by the
configuration-based queueing-control proof:

* job classes `I`;
* global configuration actions `A`;
* service vectors `μ a : I → ℝ`;
* queue-pressure dot products `qᵀ μ(a)`;
* the `ℓ₁` pressure norm for nonnegative queue vectors;
* finite nonempty action families.

The definitions deliberately use a **global action** `a : A`, not a
per-job local configuration.  A single value of `a` may encode all four
compute-scheduling quadrants:

* one job using many CPU resources;
* many jobs sharing CPU resources;
* one job using many GPUs as a gang-scheduled configuration;
* many jobs co-located on one GPU with non-additive service.
-/

noncomputable section

set_option linter.unusedSectionVars false
set_option linter.unusedVariables false

namespace Scheduleurm

open BigOperators

/-! ## Vectors and queue pressure -/

/-- A service vector indexed by job class. -/
abbrev ServiceVec (I : Type*) := I → ℝ

/-- Queue-weighted service, `qᵀ v`. -/
def dot {I : Type*} [Fintype I] (q v : ServiceVec I) : ℝ :=
  ∑ i : I, q i * v i

/-- The `ℓ₁` norm.  For nonnegative queues this is total backlog. -/
def l1 {I : Type*} [Fintype I] (q : ServiceVec I) : ℝ :=
  ∑ i : I, |q i|

/-- Pointwise nonnegativity. -/
def Nonnegative {I : Type*} (q : ServiceVec I) : Prop :=
  ∀ i : I, 0 ≤ q i

/-- For nonnegative vectors, `l1` is the plain sum. -/
lemma l1_eq_sum_of_nonnegative {I : Type*} [Fintype I]
    {q : ServiceVec I} (hq : Nonnegative q) :
    l1 q = ∑ i : I, q i := by
  unfold l1 Nonnegative at *
  refine Finset.sum_congr rfl ?_
  intro i _
  exact abs_of_nonneg (hq i)

/-- The `ℓ₁` norm is nonnegative. -/
lemma l1_nonneg {I : Type*} [Fintype I] (q : ServiceVec I) :
    0 ≤ l1 q := by
  unfold l1
  exact Finset.sum_nonneg (fun i _ => abs_nonneg (q i))

/-- Dot product is monotone in the service vector for nonnegative queues. -/
lemma dot_mono_service {I : Type*} [Fintype I]
    {q v w : ServiceVec I}
    (hq : Nonnegative q) (hvw : ∀ i, v i ≤ w i) :
    dot q v ≤ dot q w := by
  unfold dot Nonnegative at *
  apply Finset.sum_le_sum
  intro i _
  exact mul_le_mul_of_nonneg_left (hvw i) (hq i)

/-- If every coordinate changes by at most `ε`, then the queue-weighted
change is at most `ε * ||q||₁`. -/
lemma dot_le_dot_add_l1_error {I : Type*} [Fintype I]
    {q v w : ServiceVec I} {ε : ℝ}
    (hq : Nonnegative q) (hε : 0 ≤ ε)
    (hcoord : ∀ i, v i ≤ w i + ε) :
    dot q v ≤ dot q w + ε * l1 q := by
  unfold dot
  calc
    ∑ i : I, q i * v i
        ≤ ∑ i : I, q i * (w i + ε) := by
          apply Finset.sum_le_sum
          intro i _
          exact mul_le_mul_of_nonneg_left (hcoord i) (hq i)
    _ = ∑ i : I, (q i * w i + ε * q i) := by
          apply Finset.sum_congr rfl
          intro i _
          ring
    _ = ∑ i : I, q i * w i + ε * ∑ i : I, q i := by
          rw [Finset.sum_add_distrib]
          rw [Finset.mul_sum]
    _ = ∑ i : I, q i * w i + ε * l1 q := by
          rw [l1_eq_sum_of_nonnegative hq]

/-- Two service vectors within `ε` in `ℓ∞` imply the queue-weighted
one-sided bound used in candidate-set approximation. -/
lemma dot_lipschitz_linf {I : Type*} [Fintype I]
    {q v w : ServiceVec I} {ε : ℝ}
    (hq : Nonnegative q) (hε : 0 ≤ ε)
    (hcoord : ∀ i, |v i - w i| ≤ ε) :
    dot q v ≤ dot q w + ε * l1 q := by
  apply dot_le_dot_add_l1_error hq hε
  intro i
  have hi := hcoord i
  have hle : v i - w i ≤ ε := (abs_le.mp hi).2
  linarith

/-! ## Global configuration action families -/

/-- A finite nonempty set of global configuration actions. -/
structure ActionFamily (A : Type*) [DecidableEq A] where
  acts : Finset A
  nonempty : acts.Nonempty

namespace ActionFamily

variable {A : Type*} [DecidableEq A]

/-- Candidate-set containment. -/
def Subset (cand full : ActionFamily A) : Prop :=
  cand.acts ⊆ full.acts

end ActionFamily

end Scheduleurm


/-! ## Source: Scheduleurm/SupportFunction.lean -/


/-!
# Scheduleurm: support functions for configuration action sets

For a finite family of global configuration actions `A`, the support
function

`H_A(q) = max_{a ∈ A} qᵀ μ(a)`

is the MaxWeight value in queue-pressure direction `q`.  This is the
right primitive for all later theorems:

* capacity regions are characterised by support inequalities;
* candidate sets are good when their support functions approximate the
  full action set in every queue-pressure direction;
* approximate MaxWeight stability is exactly a support-function loss bound.
-/

noncomputable section

set_option linter.unusedSectionVars false
set_option linter.unusedVariables false

namespace Scheduleurm

open BigOperators

variable {I A : Type*} [Fintype I] [DecidableEq A]

/-- MaxWeight support function of a finite nonempty action family. -/
def support (F : ActionFamily A) (μ : A → ServiceVec I)
    (q : ServiceVec I) : ℝ :=
  F.acts.sup' F.nonempty (fun a => dot q (μ a))

/-- Any action in the family is bounded above by the support value. -/
lemma le_support_of_mem (F : ActionFamily A) (μ : A → ServiceVec I)
    (q : ServiceVec I) {a : A} (ha : a ∈ F.acts) :
    dot q (μ a) ≤ support F μ q := by
  unfold support
  exact Finset.le_sup' (f := fun a => dot q (μ a)) ha

/-- To upper-bound a support function, it suffices to upper-bound every
configuration action in the family. -/
lemma support_le (F : ActionFamily A) (μ : A → ServiceVec I)
    (q : ServiceVec I) {c : ℝ}
    (h : ∀ a ∈ F.acts, dot q (μ a) ≤ c) :
    support F μ q ≤ c := by
  unfold support
  exact Finset.sup'_le F.nonempty (fun a => dot q (μ a)) h

/-- Support is monotone under action-set inclusion. -/
lemma support_mono (cand full : ActionFamily A) (μ : A → ServiceVec I)
    (q : ServiceVec I) (hsub : ActionFamily.Subset cand full) :
    support cand μ q ≤ support full μ q := by
  apply support_le
  intro a ha
  exact le_support_of_mem full μ q (hsub ha)

/-- The support value is attained by at least one finite action. -/
lemma exists_mem_support_eq (F : ActionFamily A) (μ : A → ServiceVec I)
    (q : ServiceVec I) :
    ∃ a ∈ F.acts, support F μ q = dot q (μ a) := by
  unfold support
  exact F.acts.exists_mem_eq_sup' F.nonempty (fun a => dot q (μ a))

/-- A candidate action that attains candidate support is an approximate
MaxWeight action for the full family when the support gap is bounded. -/
theorem candidate_attainer_is_approx_maxweight
    (cand full : ActionFamily A) (μ : A → ServiceVec I)
    (q : ServiceVec I) {ε : ℝ}
    (hgap : support full μ q ≤ support cand μ q + ε * l1 q)
    {a : A} (ha : a ∈ cand.acts)
    (hattain : support cand μ q = dot q (μ a)) :
    support full μ q ≤ dot q (μ a) + ε * l1 q := by
  rw [hattain] at hgap
  exact hgap

end Scheduleurm


/-! ## Source: Scheduleurm/CandidateApprox.lean -/


/-!
# Scheduleurm: candidate-set structural robustness

This file formalises the BAPR-HRO insight at the configuration level:

> the candidate configuration set need not equal the full configuration
> set, provided it covers the full set in the metric that controls service.

The theorem proved here is the support-function version of the
candidate-set capacity-loss statement in `math.md`:

`H_full(q) ≤ H_cand(q) + ε ||q||₁`.

This is stronger than a per-candidate score identity because it holds in
every queue-pressure direction `q`; it is exactly the object used by
MaxWeight drift proofs.
-/

noncomputable section

set_option linter.unusedSectionVars false
set_option linter.unusedVariables false

namespace Scheduleurm

open BigOperators

variable {I A : Type*} [Fintype I] [DecidableEq A]

/-- Candidate set `cand` covers full set `full` at service-vector radius
`radius`: every full action has a candidate action whose service vector is
coordinatewise within `radius`.

The metric can encode topology/interference similarity externally; Lean only
needs the induced Lipschitz consequence on service vectors. -/
def CandidateCovers
    (full cand : ActionFamily A) (μ : A → ServiceVec I) (radius : ℝ) : Prop :=
  ∀ a ∈ full.acts, ∃ b ∈ cand.acts, ∀ i : I, |μ a i - μ b i| ≤ radius

/-- **Candidate-set structural robustness, support form.**

If each full configuration action has a candidate action within service
radius `ε`, then the candidate support function loses at most
`ε ||q||₁` in any nonnegative queue-pressure direction. -/
theorem candidate_cover_support_gap
    (full cand : ActionFamily A) (μ : A → ServiceVec I)
    {q : ServiceVec I} {ε : ℝ}
    (hq : Nonnegative q) (hε : 0 ≤ ε)
    (hcover : CandidateCovers full cand μ ε) :
    support full μ q ≤ support cand μ q + ε * l1 q := by
  apply support_le
  intro a ha
  rcases hcover a ha with ⟨b, hb, hcoord⟩
  have hdot : dot q (μ a) ≤ dot q (μ b) + ε * l1 q :=
    dot_lipschitz_linf hq hε hcoord
  have hbs : dot q (μ b) ≤ support cand μ q :=
    le_support_of_mem cand μ q hb
  linarith

/-- Candidate-set approximation condition stated directly in support
function form. -/
def SupportGapAtMost
    (full cand : ActionFamily A) (μ : A → ServiceVec I) (ε : ℝ) : Prop :=
  ∀ q : ServiceVec I, Nonnegative q →
    support full μ q ≤ support cand μ q + ε * l1 q

/-- A metric cover induces a support-function approximation guarantee. -/
theorem support_gap_from_candidate_cover
    (full cand : ActionFamily A) (μ : A → ServiceVec I) {ε : ℝ}
    (hε : 0 ≤ ε) (hcover : CandidateCovers full cand μ ε) :
    SupportGapAtMost full cand μ ε := by
  intro q hq
  exact candidate_cover_support_gap full cand μ hq hε hcover

/-- A candidate support maximizer is an `ε`-approximate full MaxWeight
action in every nonnegative queue direction. -/
theorem candidate_support_attainer_approx_full
    (full cand : ActionFamily A) (μ : A → ServiceVec I)
    {q : ServiceVec I} {ε : ℝ}
    (hq : Nonnegative q)
    (hgap : SupportGapAtMost full cand μ ε)
    {a : A} (ha : a ∈ cand.acts)
    (hattain : support cand μ q = dot q (μ a)) :
    support full μ q ≤ dot q (μ a) + ε * l1 q := by
  exact candidate_attainer_is_approx_maxweight cand full μ q (hgap q hq) ha hattain

/-! ## Metric / topology cover form -/

/-- Full actions are covered by candidate actions within radius `ρ` under
an externally supplied topology/interference distance `d`. -/
def MetricCandidateCovers
    (full cand : ActionFamily A) (d : A → A → ℝ) (ρ : ℝ) : Prop :=
  ∀ a ∈ full.acts, ∃ b ∈ cand.acts, d a b ≤ ρ

/-- Service vectors are Lipschitz with respect to an action metric.  This is
where topology and co-location interference enter the proof: if two global
configuration patterns are close in the topology/interference metric, then
their service vectors are close coordinatewise. -/
def ServiceLipschitz
    (μ : A → ServiceVec I) (d : A → A → ℝ) (L : ℝ) : Prop :=
  ∀ a b i, |μ a i - μ b i| ≤ L * d a b

/-- A metric cover plus a Lipschitz service model induces a service-vector
cover. -/
theorem candidate_cover_from_metric_lipschitz
    (full cand : ActionFamily A) (μ : A → ServiceVec I)
    (d : A → A → ℝ) {L ρ : ℝ}
    (hL : 0 ≤ L)
    (hmetric : MetricCandidateCovers full cand d ρ)
    (hlip : ServiceLipschitz μ d L) :
    CandidateCovers full cand μ (L * ρ) := by
  intro a ha
  rcases hmetric a ha with ⟨b, hb, hdist⟩
  refine ⟨b, hb, ?_⟩
  intro i
  exact le_trans (hlip a b i) (mul_le_mul_of_nonneg_left hdist hL)

/-- **Candidate-set structural robustness, metric form.**

If candidate global configurations cover the full configuration space under
a topology/interference metric, and service is `L`-Lipschitz in that metric,
then the support-function loss is at most `Lρ ||q||₁`. -/
theorem metric_cover_support_gap
    (full cand : ActionFamily A) (μ : A → ServiceVec I)
    (d : A → A → ℝ) {L ρ : ℝ}
    (hL : 0 ≤ L) (hρ : 0 ≤ ρ)
    (hmetric : MetricCandidateCovers full cand d ρ)
    (hlip : ServiceLipschitz μ d L) :
    SupportGapAtMost full cand μ (L * ρ) := by
  have hcover : CandidateCovers full cand μ (L * ρ) :=
    candidate_cover_from_metric_lipschitz full cand μ d hL hmetric hlip
  exact support_gap_from_candidate_cover full cand μ (mul_nonneg hL hρ) hcover

/-! ## Constructive finite-feature-cell candidate families -/

section FiniteFeatureCells

variable {C : Type*} [DecidableEq C]

/-- Active feature cells induced by the full action family. -/
def activeFeatureCells
    (full : ActionFamily A) (cell : A → C) : Finset C :=
  full.acts.image cell

/-- Construct a candidate family by retaining one representative for every
active feature cell.  Duplicate representatives are removed by `Finset.image`.
-/
def candidateFamilyFromFeatureCells
    (full : ActionFamily A) (cell : A → C) (rep : C → A) : ActionFamily A where
  acts := (activeFeatureCells full cell).image rep
  nonempty := (full.nonempty.image cell).image rep

/-- If each active-cell representative is a full action, the constructed
candidate family is a subset of the full family. -/
theorem candidateFamilyFromFeatureCells_subset
    (full : ActionFamily A) (cell : A → C) (rep : C → A)
    (hrep : ∀ a ∈ full.acts, rep (cell a) ∈ full.acts) :
    ActionFamily.Subset (candidateFamilyFromFeatureCells full cell rep) full := by
  intro b hb
  rcases Finset.mem_image.mp hb with ⟨c, hc, rfl⟩
  rcases Finset.mem_image.mp hc with ⟨a, ha, rfl⟩
  exact hrep a ha

/-- The number of constructed candidates is at most the number of available
feature cells. -/
theorem candidateFamilyFromFeatureCells_card_le_active
    (full : ActionFamily A) (cell : A → C) (rep : C → A) :
    (candidateFamilyFromFeatureCells full cell rep).acts.card
      ≤ (activeFeatureCells full cell).card := by
  exact Finset.card_image_le

/-- If the feature-cell type is finite, the candidate count is also bounded by
the total number of available cell labels. -/
theorem candidateFamilyFromFeatureCells_card_le
    [Fintype C]
    (full : ActionFamily A) (cell : A → C) (rep : C → A) :
    (candidateFamilyFromFeatureCells full cell rep).acts.card
      ≤ Fintype.card C := by
  change ((full.acts.image cell).image rep).card ≤ Fintype.card C
  calc
    ((full.acts.image cell).image rep).card
        ≤ (full.acts.image cell).card :=
      candidateFamilyFromFeatureCells_card_le_active full cell rep
    _ ≤ (Finset.univ : Finset C).card :=
      Finset.card_le_card (Finset.subset_univ _)
    _ = Fintype.card C := Finset.card_univ

/-- A representative-radius certificate makes the finite-cell construction a
metric candidate cover. -/
theorem featureCellCandidate_metricCover
    (full : ActionFamily A) (cell : A → C) (rep : C → A)
    (d : A → A → ℝ) (ρ : ℝ)
    (hradius : ∀ a ∈ full.acts, d a (rep (cell a)) ≤ ρ) :
    MetricCandidateCovers full
      (candidateFamilyFromFeatureCells full cell rep) d ρ := by
  intro a ha
  refine ⟨rep (cell a), ?_, hradius a ha⟩
  apply Finset.mem_image.mpr
  refine ⟨cell a, ?_, rfl⟩
  exact Finset.mem_image.mpr ⟨a, ha, rfl⟩

/-- **Constructive finite-cell candidate certificate.**

A finite feature partition, one valid representative per active cell, a
representative-radius certificate, and service Lipschitzness jointly give:

* candidate containment in the full action family;
* candidate cardinality at most the number of feature cells;
* support-function loss at most `L * ρ` in every nonnegative queue direction.
-/
theorem finite_feature_cell_candidate_certificate
    (full : ActionFamily A) (cell : A → C) (rep : C → A)
    (μ : A → ServiceVec I) (d : A → A → ℝ)
    {L ρ : ℝ}
    (hL : 0 ≤ L) (hρ : 0 ≤ ρ)
    (hrep : ∀ a ∈ full.acts, rep (cell a) ∈ full.acts)
    (hradius : ∀ a ∈ full.acts, d a (rep (cell a)) ≤ ρ)
    (hlip : ServiceLipschitz μ d L) :
    ActionFamily.Subset
        (candidateFamilyFromFeatureCells full cell rep) full ∧
      (candidateFamilyFromFeatureCells full cell rep).acts.card
        ≤ (activeFeatureCells full cell).card ∧
      SupportGapAtMost full
        (candidateFamilyFromFeatureCells full cell rep) μ (L * ρ) := by
  refine ⟨candidateFamilyFromFeatureCells_subset full cell rep hrep, ?_, ?_⟩
  · exact candidateFamilyFromFeatureCells_card_le_active full cell rep
  · exact metric_cover_support_gap full
      (candidateFamilyFromFeatureCells full cell rep) μ d
      hL hρ (featureCellCandidate_metricCover full cell rep d ρ hradius) hlip

end FiniteFeatureCells

end Scheduleurm


/-! ## Source: Scheduleurm/CapacityRegion.lean -/


/-!
# Scheduleurm: configuration capacity region

This file formalises the capacity-region layer of `math.md`.

For a finite global action family `𝒜`, a stationary randomized scheduler is
a distribution over actions in `𝒜`.  Its average service vector is the
convex combination of the corresponding service vectors.  A load vector
`λ` lies in the capacity region with uniform slack `δ` when some stationary
mix delivers at least `λ_i + δ` service to every job class `i`.

The main nontrivial lemma proves the standard support-function consequence:

`λ + δ·1 ∈ conv{μ(a)}` implies

`qᵀ λ + δ ||q||₁ ≤ H_𝒜(q)` for every nonnegative queue vector `q`.

This is the exact bridge from capacity-region geometry to MaxWeight drift.
-/

noncomputable section

set_option linter.unusedSectionVars false
set_option linter.unusedVariables false

namespace Scheduleurm

open BigOperators

variable {I A : Type*} [Fintype I] [DecidableEq A]

/-- A stationary randomized mix over a finite action family. -/
structure StationaryMix (F : ActionFamily A) where
  weight : A → ℝ
  nonneg_on : ∀ a ∈ F.acts, 0 ≤ weight a
  sum_one : (∑ a ∈ F.acts, weight a) = 1

/-- Average service vector induced by a stationary mix. -/
def mixedService (F : ActionFamily A) (μ : A → ServiceVec I)
    (x : StationaryMix F) : ServiceVec I :=
  fun i => ∑ a ∈ F.acts, x.weight a * μ a i

/-- Capacity-region membership with a uniform per-class slack `δ`. -/
def InCapacityWithSlack (F : ActionFamily A) (μ : A → ServiceVec I)
    (lam : ServiceVec I) (δ : ℝ) : Prop :=
  ∃ x : StationaryMix F, ∀ i : I, lam i + δ ≤ mixedService F μ x i

/-- Downward-closed capacity-region membership with slack.  This is the
queueing-control capacity object: the offered load may be coordinatewise
below a stationary average service vector because unused service can be
wasted. -/
def InDownwardCapacityWithSlack
    (F : ActionFamily A) (μ : A → ServiceVec I)
    (lam : ServiceVec I) (δ : ℝ) : Prop :=
  ∃ x : StationaryMix F, ∀ i : I, lam i + δ ≤ mixedService F μ x i

/-- Zero-slack downward-closed capacity region. -/
def InDownwardCapacityRegion
    (F : ActionFamily A) (μ : A → ServiceVec I)
    (lam : ServiceVec I) : Prop :=
  InDownwardCapacityWithSlack F μ lam 0

/-- The existing slack definition is already the downward-closed capacity
definition; this theorem fixes the paper notation so the text does not
incorrectly identify capacity with the convex hull boundary only. -/
theorem inCapacityWithSlack_iff_downwardCapacityWithSlack
    (F : ActionFamily A) (μ : A → ServiceVec I)
    (lam : ServiceVec I) (δ : ℝ) :
    InCapacityWithSlack F μ lam δ ↔
      InDownwardCapacityWithSlack F μ lam δ := by
  rfl

/-- Downward closure: if a load with slack is supportable, any coordinatewise
smaller load is supportable with the same slack. -/
theorem downward_capacity_monotone
    (F : ActionFamily A) (μ : A → ServiceVec I)
    {lam lam' : ServiceVec I} {δ : ℝ}
    (hle : ∀ i : I, lam' i ≤ lam i)
    (hcap : InDownwardCapacityWithSlack F μ lam δ) :
    InDownwardCapacityWithSlack F μ lam' δ := by
  rcases hcap with ⟨x, hx⟩
  refine ⟨x, ?_⟩
  intro i
  have hi := hx i
  have hle_i := hle i
  linarith

/-- Same object, named from the operational perspective. -/
def StabilizableByStationaryMix (F : ActionFamily A) (μ : A → ServiceVec I)
    (lam : ServiceVec I) (δ : ℝ) : Prop :=
  ∃ x : StationaryMix F, ∀ i : I, lam i + δ ≤ mixedService F μ x i

/-- **Configuration capacity-region characterisation.**

In the finite-action model, being in the capacity region with slack is
equivalent to being supportable by a stationary randomized mix over global
configuration actions.  The equivalence is definitional; the nontrivial
support-function consequence is proved below. -/
theorem configuration_capacity_region_characterization
    (F : ActionFamily A) (μ : A → ServiceVec I)
    (lam : ServiceVec I) (δ : ℝ) :
    InCapacityWithSlack F μ lam δ ↔
      StabilizableByStationaryMix F μ lam δ := by
  rfl

/-- Dot product against a mixed service vector equals the corresponding
weighted sum of per-action dot products. -/
lemma dot_mixedService_eq
    (F : ActionFamily A) (μ : A → ServiceVec I)
    (x : StationaryMix F) (q : ServiceVec I) :
    dot q (mixedService F μ x) =
      ∑ a ∈ F.acts, x.weight a * dot q (μ a) := by
  unfold dot mixedService
  calc
    ∑ i : I, q i * (∑ a ∈ F.acts, x.weight a * μ a i)
        = ∑ i : I, ∑ a ∈ F.acts, q i * (x.weight a * μ a i) := by
          apply Finset.sum_congr rfl
          intro i _
          rw [Finset.mul_sum]
    _ = ∑ a ∈ F.acts, ∑ i : I, q i * (x.weight a * μ a i) := by
          rw [Finset.sum_comm]
    _ = ∑ a ∈ F.acts, x.weight a * ∑ i : I, q i * μ a i := by
          apply Finset.sum_congr rfl
          intro a _
          rw [Finset.mul_sum]
          apply Finset.sum_congr rfl
          intro i _
          ring

/-- A stationary mix cannot exceed the support function in direction `q`. -/
lemma dot_mixedService_le_support
    (F : ActionFamily A) (μ : A → ServiceVec I)
    (x : StationaryMix F) (q : ServiceVec I) :
    dot q (mixedService F μ x) ≤ support F μ q := by
  rw [dot_mixedService_eq F μ x q]
  calc
    ∑ a ∈ F.acts, x.weight a * dot q (μ a)
        ≤ ∑ a ∈ F.acts, x.weight a * support F μ q := by
          apply Finset.sum_le_sum
          intro a ha
          exact mul_le_mul_of_nonneg_left
            (le_support_of_mem F μ q ha) (x.nonneg_on a ha)
    _ = (∑ a ∈ F.acts, x.weight a) * support F μ q := by
          rw [Finset.sum_mul]
    _ = support F μ q := by
          rw [x.sum_one]
          ring

/-- Expanding `qᵀ(λ + δ·1)` gives `qᵀλ + δ ||q||₁` for nonnegative queues. -/
lemma dot_add_uniform_slack
    {q lam : ServiceVec I} {δ : ℝ} (hq : Nonnegative q) :
    dot q (fun i => lam i + δ) = dot q lam + δ * l1 q := by
  unfold dot
  rw [l1_eq_sum_of_nonnegative hq]
  calc
    ∑ i : I, q i * (lam i + δ)
        = ∑ i : I, (q i * lam i + δ * q i) := by
          apply Finset.sum_congr rfl
          intro i _
          ring
    _ = ∑ i : I, q i * lam i + δ * ∑ i : I, q i := by
          rw [Finset.sum_add_distrib]
          rw [Finset.mul_sum]

/-- If a stationary mix has uniform slack `δ`, then every nonnegative
queue-pressure direction has support slack `δ ||q||₁`. -/
theorem capacity_slack_implies_support_slack
    (F : ActionFamily A) (μ : A → ServiceVec I)
    {lam q : ServiceVec I} {δ : ℝ}
    (hq : Nonnegative q)
    (hcap : InCapacityWithSlack F μ lam δ) :
    dot q lam + δ * l1 q ≤ support F μ q := by
  rcases hcap with ⟨x, hx⟩
  have hcoord : ∀ i : I, (fun j => lam j + δ) i ≤ mixedService F μ x i := hx
  have hdot : dot q (fun i => lam i + δ) ≤ dot q (mixedService F μ x) :=
    dot_mono_service hq hcoord
  have hmix : dot q (mixedService F μ x) ≤ support F μ q :=
    dot_mixedService_le_support F μ x q
  rw [dot_add_uniform_slack hq] at hdot
  exact le_trans hdot hmix

/-- Downward-closed capacity slack has the same MaxWeight support-function
consequence. -/
theorem downward_capacity_slack_implies_support_slack
    (F : ActionFamily A) (μ : A → ServiceVec I)
    {lam q : ServiceVec I} {δ : ℝ}
    (hq : Nonnegative q)
    (hcap : InDownwardCapacityWithSlack F μ lam δ) :
    dot q lam + δ * l1 q ≤ support F μ q := by
  exact capacity_slack_implies_support_slack F μ hq hcap

/-- **Candidate capacity-loss theorem.**

If the full action family supports load `lam` with slack `δ`, and the
candidate action family has support-function loss at most `ε`, then the
candidate family preserves slack `δ - ε` in every nonnegative
queue-pressure direction.  This is the capacity-region version of the
BAPR-HRO "keep the structure, rerank candidates" principle. -/
theorem candidate_capacity_slack_loss
    (full cand : ActionFamily A) (μ : A → ServiceVec I)
    {lam q : ServiceVec I} {δ ε : ℝ}
    (hq : Nonnegative q)
    (hcap : InCapacityWithSlack full μ lam δ)
    (hgap : SupportGapAtMost full cand μ ε) :
    dot q lam + (δ - ε) * l1 q ≤ support cand μ q := by
  have hfull : dot q lam + δ * l1 q ≤ support full μ q :=
    capacity_slack_implies_support_slack full μ hq hcap
  have hgapq : support full μ q ≤ support cand μ q + ε * l1 q :=
    hgap q hq
  have hchain : dot q lam + δ * l1 q ≤ support cand μ q + ε * l1 q :=
    le_trans hfull hgapq
  linarith

end Scheduleurm


/-! ## Source: Scheduleurm/MaxWeightDrift.lean -/


/-!
# Scheduleurm: approximate MaxWeight Lyapunov drift

This file proves the queueing-control core from `math.md`.

The result is stated in two layers:

1. A standard one-step quadratic Lyapunov inequality for queue dynamics
   `Q⁺ = [Q - S]⁺ + A`.
2. An approximate-MaxWeight negative drift theorem: if
   * the arrival vector is inside the full capacity region with slack `δ`;
   * the policy loses at most `ε ||Q||₁` relative to full MaxWeight;
   * `δ > ε`;
   then the drift has the negative backlog term
   `-(δ - ε)||Q||₁`, up to the bounded second-moment and switching/risk cost.

The theorem is intentionally written for a generic chosen service vector `S`.
In applications, `S` is either the actual service of the selected
configuration or a robust lower service vector after BAPR/BAPR-HRO scoring.
-/

noncomputable section

set_option linter.unusedSectionVars false
set_option linter.unusedVariables false

namespace Scheduleurm

open BigOperators

variable {I A : Type*} [Fintype I] [DecidableEq A]

/-- Scalar queue update `[q - s]^+ + a`. -/
def queueStepScalar (q a s : ℝ) : ℝ :=
  max (q - s) 0 + a

/-- Vector queue update. -/
def queueStep (Q Arr Serv : ServiceVec I) : ServiceVec I :=
  fun i => queueStepScalar (Q i) (Arr i) (Serv i)

/-- Quadratic Lyapunov function `V(Q)=1/2∑Q_i²`. -/
def Lyapunov (Q : ServiceVec I) : ℝ :=
  ∑ i : I, (Q i)^2 / 2

/-- Per-step second-order term in the standard queue drift inequality. -/
def secondOrderTerm (Arr Serv : ServiceVec I) : ℝ :=
  ∑ i : I, ((Arr i)^2 + (Serv i)^2) / 2

/-- Scalar form of the standard queueing inequality:

`([q-s]^+ + a)^2 - q^2 ≤ a^2 + s^2 + 2q(a-s)`.
-/
lemma queue_step_square_bound
    {q a s : ℝ} (hq : 0 ≤ q) (ha : 0 ≤ a) (hs : 0 ≤ s) :
    (queueStepScalar q a s)^2 - q^2
      ≤ s^2 + a^2 + 2 * q * (a - s) := by
  unfold queueStepScalar
  by_cases hle : q - s ≤ 0
  · have hmax : max (q - s) 0 = 0 := max_eq_right hle
    rw [hmax]
    have hsq : 0 ≤ (q - s)^2 := sq_nonneg (q - s)
    have hqa : 0 ≤ 2 * q * a := by nlinarith [mul_nonneg hq ha]
    nlinarith [hsq, hqa]
  · have hlt : 0 < q - s := lt_of_not_ge hle
    have hge : 0 ≤ q - s := le_of_lt hlt
    have hmax : max (q - s) 0 = q - s := max_eq_left hge
    rw [hmax]
    have has : 0 ≤ 2 * a * s := by nlinarith [mul_nonneg ha hs]
    nlinarith [has]

/-- Divided-by-two scalar drift inequality. -/
lemma queue_step_half_square_bound
    {q a s : ℝ} (hq : 0 ≤ q) (ha : 0 ≤ a) (hs : 0 ≤ s) :
    (queueStepScalar q a s)^2 / 2 - q^2 / 2
      ≤ (a^2 + s^2) / 2 + q * a - q * s := by
  have h := queue_step_square_bound hq ha hs
  nlinarith

/-- Vector Lyapunov drift inequality. -/
theorem lyapunov_queue_step_bound
    {Q Arr Serv : ServiceVec I}
    (hQ : Nonnegative Q) (hArr : Nonnegative Arr) (hServ : Nonnegative Serv) :
    Lyapunov (queueStep Q Arr Serv) - Lyapunov Q
      ≤ secondOrderTerm Arr Serv + dot Q Arr - dot Q Serv := by
  unfold Lyapunov secondOrderTerm dot queueStep
  calc
    (∑ i : I, (queueStepScalar (Q i) (Arr i) (Serv i))^2 / 2)
        - ∑ i : I, (Q i)^2 / 2
        = ∑ i : I,
            ((queueStepScalar (Q i) (Arr i) (Serv i))^2 / 2 - (Q i)^2 / 2) := by
          rw [Finset.sum_sub_distrib]
    _ ≤ ∑ i : I, (((Arr i)^2 + (Serv i)^2) / 2
            + Q i * Arr i - Q i * Serv i) := by
          apply Finset.sum_le_sum
          intro i _
          exact queue_step_half_square_bound (hQ i) (hArr i) (hServ i)
    _ = (∑ i : I, ((Arr i)^2 + (Serv i)^2) / 2)
          + (∑ i : I, Q i * Arr i) - (∑ i : I, Q i * Serv i) := by
          rw [Finset.sum_sub_distrib, Finset.sum_add_distrib]

/-- Approximate MaxWeight drift inequality.

`cost` collects switching, migration, rollback and tail-risk penalties.
The theorem keeps this term explicit rather than hiding it in the service
model, matching the objective in `math.md`:

`Qᵀ μ(a) - K(a_prev,a) - R(a)`.
-/
theorem approximate_maxWeight_negative_drift
    (full : ActionFamily A) (μ : A → ServiceVec I)
    {Q lam Serv : ServiceVec I} {δ ε B cost : ℝ}
    (hQ : Nonnegative Q)
    (hcap : InCapacityWithSlack full μ lam δ)
    (hApprox : support full μ Q ≤ dot Q Serv + ε * l1 Q + cost)
    (hSecond : secondOrderTerm lam Serv ≤ B) :
    secondOrderTerm lam Serv + dot Q lam - dot Q Serv
      ≤ B + cost - (δ - ε) * l1 Q := by
  have hSlack : dot Q lam + δ * l1 Q ≤ support full μ Q :=
    capacity_slack_implies_support_slack full μ hQ hcap
  have hService : dot Q lam + δ * l1 Q ≤ dot Q Serv + ε * l1 Q + cost :=
    le_trans hSlack hApprox
  have hPressure : dot Q lam - dot Q Serv
      ≤ cost - (δ - ε) * l1 Q := by
    linarith
  linarith

/-- Full one-step drift theorem obtained by combining the queue dynamics
inequality with approximate MaxWeight. -/
theorem approximate_maxWeight_lyapunov_drift
    (full : ActionFamily A) (μ : A → ServiceVec I)
    {Q lam Serv : ServiceVec I} {δ ε B cost : ℝ}
    (hQ : Nonnegative Q) (hlam : Nonnegative lam) (hServ : Nonnegative Serv)
    (hcap : InCapacityWithSlack full μ lam δ)
    (hApprox : support full μ Q ≤ dot Q Serv + ε * l1 Q + cost)
    (hSecond : secondOrderTerm lam Serv ≤ B) :
    Lyapunov (queueStep Q lam Serv) - Lyapunov Q
      ≤ B + cost - (δ - ε) * l1 Q := by
  have hQueue := lyapunov_queue_step_bound (Q := Q) (Arr := lam) (Serv := Serv)
    hQ hlam hServ
  have hMW := approximate_maxWeight_negative_drift full μ hQ hcap hApprox hSecond
  exact le_trans hQueue hMW

end Scheduleurm


/-! ## Source: Scheduleurm/RegimeBelief.lean -/


/-!
# Scheduleurm: hidden regime belief layer

This file formalises the BAPR layer from `math.md`.

The scheduler may maintain a joint BOCD-style belief over
`(run-length, regime)` pairs, but the queueing-control action value only
depends on the **regime marginal**.  This mirrors the earlier BAMOR joint
belief result, now for configuration-action service vectors rather than
Bellman operators.
-/

noncomputable section

set_option linter.unusedSectionVars false
set_option linter.unusedVariables false

namespace Scheduleurm

open BigOperators

variable {I A H Z : Type*}
  [Fintype I] [DecidableEq A] [Fintype H] [Fintype Z]

/-- Marginal over the hidden hardware/workload regime. -/
def marginalRegime (b : H × Z → ℝ) (z : Z) : ℝ :=
  ∑ h : H, b (h, z)

/-- Joint belief validity: nonnegative and total mass one. -/
def JointBeliefValid (b : H × Z → ℝ) : Prop :=
  (∀ hz : H × Z, 0 ≤ b hz) ∧ (∑ hz : H × Z, b hz = 1)

/-- Regime-marginal belief validity. -/
def RegimeBeliefValid (β : Z → ℝ) : Prop :=
  (∀ z : Z, 0 ≤ β z) ∧ (∑ z : Z, β z = 1)

/-- The regime marginal of a valid joint belief is a valid regime belief. -/
theorem marginalRegime_valid {b : H × Z → ℝ}
    (hb : JointBeliefValid b) :
    RegimeBeliefValid (marginalRegime b) := by
  rcases hb with ⟨hb_nn, hb_sum⟩
  constructor
  · intro z
    unfold marginalRegime
    exact Finset.sum_nonneg (fun h _ => hb_nn (h, z))
  · unfold marginalRegime
    rw [← hb_sum]
    rw [Fintype.sum_prod_type]
    exact Finset.sum_comm

/-- Belief-weighted service when the belief is already marginalised over regimes. -/
def beliefWeightedService
    (β : Z → ℝ) (μ : Z → A → ServiceVec I) (a : A) : ServiceVec I :=
  fun i => ∑ z : Z, β z * μ z a i

/-- Belief-weighted service from the full `(run-length, regime)` joint belief. -/
def jointBeliefWeightedService
    (b : H × Z → ℝ) (μ : Z → A → ServiceVec I) (a : A) : ServiceVec I :=
  fun i => ∑ hz : H × Z, b hz * μ hz.2 a i

/-- Joint belief and regime-marginal belief induce exactly the same
configuration-action service vector. -/
theorem jointBeliefWeightedService_eq_marginal
    (b : H × Z → ℝ) (μ : Z → A → ServiceVec I) (a : A) :
    jointBeliefWeightedService b μ a =
      beliefWeightedService (marginalRegime b) μ a := by
  funext i
  unfold jointBeliefWeightedService beliefWeightedService marginalRegime
  rw [Fintype.sum_prod_type]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro z _
  simp [Finset.sum_mul]

/-- If `lower` is a pointwise lower bound for every regime-specific service
vector, then it is also a lower bound for the belief-weighted service. -/
theorem lower_service_le_beliefWeighted
    {β : Z → ℝ} (hβ : RegimeBeliefValid β)
    (μ : Z → A → ServiceVec I) (lower : A → ServiceVec I)
    (a : A)
    (hlower : ∀ z i, lower a i ≤ μ z a i) :
    ∀ i : I, lower a i ≤ beliefWeightedService β μ a i := by
  intro i
  rcases hβ with ⟨hβ_nn, hβ_sum⟩
  calc
    lower a i
        = (∑ z : Z, β z) * lower a i := by
          rw [hβ_sum]
          ring
    _ = ∑ z : Z, β z * lower a i := by
          rw [Finset.sum_mul]
    _ ≤ ∑ z : Z, β z * μ z a i := by
          apply Finset.sum_le_sum
          intro z _
          exact mul_le_mul_of_nonneg_left (hlower z i) (hβ_nn z)

/-- Robust service lower bounds can be fed into MaxWeight in place of the
unknown true regime service: using a lower service never overstates the
belief-weighted service in any nonnegative queue direction. -/
theorem dot_lower_service_le_beliefWeighted
    {β : Z → ℝ} (hβ : RegimeBeliefValid β)
    (μ : Z → A → ServiceVec I) (lower : A → ServiceVec I)
    {q : ServiceVec I} (hq : Nonnegative q) (a : A)
    (hlower : ∀ z i, lower a i ≤ μ z a i) :
    dot q (lower a) ≤ dot q (beliefWeightedService β μ a) := by
  apply dot_mono_service hq
  exact lower_service_le_beliefWeighted hβ μ lower a hlower

end Scheduleurm


/-! ## Source: Scheduleurm/SweetSpot.lean -/


/-!
# Scheduleurm: single-resource sweet spot and admission threshold

This file formalises Theorem F from `math.md` for one shared resource
(for example, one GPU).  The total service curve `g n` may rise while
co-location improves aggregate throughput, and then fall after bandwidth
or scheduling interference dominates.

The theorem states the exact admission-threshold structure:

* before the sweet spot `n*`, the marginal gain of adding one job is positive;
* at or after `n*`, the marginal gain is nonpositive;
* therefore "accept iff marginal goodput is positive" is equivalent to
  the threshold rule `n < n*`.
-/

noncomputable section

set_option linter.unusedSectionVars false
set_option linter.unusedVariables false

namespace Scheduleurm

/-- Marginal goodput of adding the `(n+1)`-st co-located task. -/
def marginalGoodput (g : ℕ → ℝ) (n : ℕ) : ℝ :=
  g (n + 1) - g n

/-- Strictly rising before the sweet spot. -/
def StrictlyRisesBefore (g : ℕ → ℝ) (nstar : ℕ) : Prop :=
  ∀ n : ℕ, n < nstar → g n < g (n + 1)

/-- Nonincreasing at and after the sweet spot. -/
def NonincreasingAfter (g : ℕ → ℝ) (nstar : ℕ) : Prop :=
  ∀ n : ℕ, nstar ≤ n → g (n + 1) ≤ g n

/-- Threshold admission rule for a unimodal shared-resource curve. -/
def thresholdAdmission (nstar n : ℕ) : Prop :=
  n < nstar

/-- Marginal-goodput admission rule. -/
def marginalAdmission (g : ℕ → ℝ) (n : ℕ) : Prop :=
  0 < marginalGoodput g n

/-- The left side of a unimodal service curve has positive marginal goodput. -/
lemma positive_marginal_before_sweet_spot
    {g : ℕ → ℝ} {nstar n : ℕ}
    (hrise : StrictlyRisesBefore g nstar)
    (hn : n < nstar) :
    0 < marginalGoodput g n := by
  unfold marginalGoodput
  have h := hrise n hn
  linarith

/-- The right side of a unimodal service curve has nonpositive marginal goodput. -/
lemma nonpositive_marginal_after_sweet_spot
    {g : ℕ → ℝ} {nstar n : ℕ}
    (hfall : NonincreasingAfter g nstar)
    (hn : nstar ≤ n) :
    marginalGoodput g n ≤ 0 := by
  unfold marginalGoodput
  have h := hfall n hn
  linarith

/-- **Sweet spot / admission threshold theorem.**

For a service curve that strictly rises before `n*` and is nonincreasing
from `n*` onward, the marginal-goodput admission rule is exactly the
threshold rule `n < n*`. -/
theorem sweet_spot_threshold_admission
    {g : ℕ → ℝ} {nstar n : ℕ}
    (hrise : StrictlyRisesBefore g nstar)
    (hfall : NonincreasingAfter g nstar) :
    thresholdAdmission nstar n ↔ marginalAdmission g n := by
  constructor
  · intro hn
    exact positive_marginal_before_sweet_spot hrise hn
  · intro hpos
    by_contra hnot
    have hge : nstar ≤ n := Nat.le_of_not_gt hnot
    have hnonpos := nonpositive_marginal_after_sweet_spot hfall hge
    unfold marginalAdmission at hpos
    linarith

/-! ## Diminishing returns version for heterogeneous co-location -/

variable {J : Type*} [DecidableEq J]

/-- Marginal value of adding job `j` to co-location set `S`. -/
def setMarginalGoodput (g : Finset J → ℝ) (S : Finset J) (j : J) : ℝ :=
  g (insert j S) - g S

/-- Diminishing returns / submodular co-location service curve. -/
def DiminishingReturns (g : Finset J → ℝ) : Prop :=
  ∀ ⦃S T : Finset J⦄ ⦃j : J⦄,
    S ⊆ T → j ∉ T →
      setMarginalGoodput g T j ≤ setMarginalGoodput g S j

/-- If adding `j` to a larger co-location set has positive marginal goodput,
then adding it to any smaller eligible set also has positive marginal goodput.
This is the formal monotonicity behind greedy admission under diminishing
returns. -/
theorem positive_large_set_marginal_implies_positive_smaller
    {g : Finset J → ℝ} (hsubmod : DiminishingReturns g)
    {S T : Finset J} {j : J}
    (hST : S ⊆ T) (hjT : j ∉ T)
    (hpos : 0 < setMarginalGoodput g T j) :
    0 < setMarginalGoodput g S j := by
  have hle := hsubmod hST hjT
  linarith

end Scheduleurm


/-! ## Source: Scheduleurm/Learning.lean -/


/-!
# Scheduleurm: unknown service rates and lower-confidence service

This file formalises the learning layer from Theorem D in `math.md`.

When the true configuration service `μ` is unknown, the scheduler may use
an estimator `μhat` with confidence radius `rad`.  The lower-confidence
service

`LCB(a,i) = μhat(a,i) - rad(a)`

is pessimistic, and if the confidence event holds, replacing true service
by LCB costs at most `2ε ||q||₁` in support value whenever all radii are
bounded by `ε`.

This is the estimation-error component in the decomposition

`ε_total = ε_candidate + ε_score + ε_regime + ε_switch`.
-/

noncomputable section

set_option linter.unusedSectionVars false
set_option linter.unusedVariables false

namespace Scheduleurm

open BigOperators

variable {I A : Type*} [Fintype I] [DecidableEq A]

/-- Lower-confidence service vector for action `a`. -/
def lcbService (μhat : A → ServiceVec I) (rad : A → ℝ) :
    A → ServiceVec I :=
  fun a i => μhat a i - rad a

/-- The usual confidence event for service-rate learning. -/
def ServiceConfidenceEvent
    (F : ActionFamily A) (μtrue μhat : A → ServiceVec I) (rad : A → ℝ) : Prop :=
  ∀ a ∈ F.acts, ∀ i : I, |μhat a i - μtrue a i| ≤ rad a

/-- Radius bounded uniformly over the relevant action family. -/
def RadiusBounded
    (F : ActionFamily A) (rad : A → ℝ) (ε : ℝ) : Prop :=
  ∀ a ∈ F.acts, 0 ≤ rad a ∧ rad a ≤ ε

/-- Under the confidence event, the LCB service is a valid lower bound on
true service. -/
theorem lcbService_le_true
    (F : ActionFamily A) (μtrue μhat : A → ServiceVec I) (rad : A → ℝ)
    (hconf : ServiceConfidenceEvent F μtrue μhat rad) :
    ∀ a ∈ F.acts, ∀ i : I, lcbService μhat rad a i ≤ μtrue a i := by
  intro a ha i
  unfold lcbService
  have h := hconf a ha i
  have hle : μhat a i - μtrue a i ≤ rad a := (abs_le.mp h).2
  linarith

/-- Under confidence and a uniform radius bound, true service is at most
LCB service plus `2ε` in every coordinate. -/
theorem true_le_lcbService_add_two_radius
    (F : ActionFamily A) (μtrue μhat : A → ServiceVec I) (rad : A → ℝ)
    {ε : ℝ}
    (hconf : ServiceConfidenceEvent F μtrue μhat rad)
    (hrad : RadiusBounded F rad ε) :
    ∀ a ∈ F.acts, ∀ i : I,
      μtrue a i ≤ lcbService μhat rad a i + 2 * ε := by
  intro a ha i
  unfold lcbService
  have h := hconf a ha i
  have hrad_a := hrad a ha
  have h1 : μtrue a i ≤ μhat a i + rad a := by
    have hleft : -(rad a) ≤ μhat a i - μtrue a i := (abs_le.mp h).1
    linarith
  have h2 : rad a ≤ ε := hrad_a.2
  linarith

/-- **Learning/estimation support loss.**

If the service estimator is correct within radius `rad`, and every active
radius is at most `ε`, then the true support function is bounded by the
LCB support function plus `2ε ||q||₁`. -/
theorem true_support_le_lcb_support_plus_estimation_error
    (F : ActionFamily A) (μtrue μhat : A → ServiceVec I) (rad : A → ℝ)
    {q : ServiceVec I} {ε : ℝ}
    (hq : Nonnegative q) (hε : 0 ≤ ε)
    (hconf : ServiceConfidenceEvent F μtrue μhat rad)
    (hrad : RadiusBounded F rad ε) :
    support F μtrue q ≤ support F (lcbService μhat rad) q + (2 * ε) * l1 q := by
  apply support_le
  intro a ha
  have hcoord : ∀ i : I, μtrue a i ≤ lcbService μhat rad a i + 2 * ε :=
    true_le_lcbService_add_two_radius F μtrue μhat rad hconf hrad a ha
  have hdot : dot q (μtrue a)
      ≤ dot q (lcbService μhat rad a) + (2 * ε) * l1 q :=
    dot_le_dot_add_l1_error hq (by nlinarith [hε]) hcoord
  have hs : dot q (lcbService μhat rad a)
      ≤ support F (lcbService μhat rad) q :=
    le_support_of_mem F (lcbService μhat rad) q ha
  linarith

end Scheduleurm


/-! ## Source: Scheduleurm/DiagonalScaling.lean -/


/-!
# Scheduleurm: diagonal service scaling for heterogeneous workloads

The original learning theorem uses one absolute estimation radius `ε` for all
job classes.  That is too coarse for heterogeneous fabrics: a small relative
error on a high-throughput LLM/CNN profile can dominate the same absolute
budget as a CPU/control workload.

This file proves the dimension-aware replacement used by the OR submission
artifact.  Each coordinate `i` has a service scale `scale i`; service
estimation error is charged as `ε * scale i`, and queue pressure is measured by
the corresponding scaled backlog

`Σ_i scale_i Q_i`.

When the scales are uniformly bounded by `C`, the theorem reduces back to the
ordinary `ℓ₁` drift theorem with an effective loss `ε C`.
-/

noncomputable section

set_option linter.unusedSectionVars false
set_option linter.unusedVariables false

namespace Scheduleurm

open BigOperators

variable {I A : Type*} [Fintype I] [DecidableEq A]

/-! ## Coordinate-scaled pressure -/

/-- Scaled queue pressure `Σ_i scale_i Q_i`.

For normalized service units the intended choice is `scale_i = s_i`, where
`s_i` is the service-unit normalizer for workload class `i`. -/
def weightedPressure (scale q : ServiceVec I) : ℝ :=
  ∑ i : I, scale i * q i

/-- Coordinate-wise service error bounded by `ε * scale_i` yields a
queue-pressure error bounded by `ε` times scaled backlog. -/
lemma dot_le_dot_add_weighted_error
    {q v w scale : ServiceVec I} {ε : ℝ}
    (hq : Nonnegative q) (hε : 0 ≤ ε) (hscale : Nonnegative scale)
    (hcoord : ∀ i : I, v i ≤ w i + ε * scale i) :
    dot q v ≤ dot q w + ε * weightedPressure scale q := by
  unfold dot weightedPressure
  calc
    ∑ i : I, q i * v i
        ≤ ∑ i : I, q i * (w i + ε * scale i) := by
          apply Finset.sum_le_sum
          intro i _
          exact mul_le_mul_of_nonneg_left (hcoord i) (hq i)
    _ = ∑ i : I, (q i * w i + ε * (scale i * q i)) := by
          apply Finset.sum_congr rfl
          intro i _
          ring
    _ = ∑ i : I, q i * w i + ε * ∑ i : I, scale i * q i := by
          rw [Finset.sum_add_distrib]
          rw [Finset.mul_sum]

/-- If all service scales are bounded by `C`, scaled backlog is bounded by
`C ||Q||₁`. -/
lemma weightedPressure_le_const_l1
    {scale q : ServiceVec I} {C : ℝ}
    (hq : Nonnegative q) (hscale_bound : ∀ i : I, scale i ≤ C) :
    weightedPressure scale q ≤ C * l1 q := by
  unfold weightedPressure
  calc
    ∑ i : I, scale i * q i
        ≤ ∑ i : I, C * q i := by
          apply Finset.sum_le_sum
          intro i _
          exact mul_le_mul_of_nonneg_right (hscale_bound i) (hq i)
    _ = C * ∑ i : I, q i := by
          rw [Finset.mul_sum]
    _ = C * l1 q := by
          rw [l1_eq_sum_of_nonnegative hq]

/-! ## Support-function consequences -/

/-- Coordinate-scaled support loss.  This is the normalized-service analogue
of `true_support_le_lcb_support_plus_estimation_error`. -/
theorem coordinate_scaled_support_loss
    (F : ActionFamily A) (μtrue lower : A → ServiceVec I)
    (scale : ServiceVec I)
    {q : ServiceVec I} {ε : ℝ}
    (hq : Nonnegative q) (hε : 0 ≤ ε) (hscale : Nonnegative scale)
    (hcoord : ∀ a ∈ F.acts, ∀ i : I,
      μtrue a i ≤ lower a i + ε * scale i) :
    support F μtrue q
      ≤ support F lower q + ε * weightedPressure scale q := by
  apply support_le
  intro a ha
  have hdot : dot q (μtrue a)
      ≤ dot q (lower a) + ε * weightedPressure scale q :=
    dot_le_dot_add_weighted_error hq hε hscale (hcoord a ha)
  have hs : dot q (lower a) ≤ support F lower q :=
    le_support_of_mem F lower q ha
  linarith

/-- Coordinate-scaled support loss converted back to the ordinary `ℓ₁`
backlog norm by a uniform bound on the scale vector. -/
theorem coordinate_scaled_support_loss_l1
    (F : ActionFamily A) (μtrue lower : A → ServiceVec I)
    (scale : ServiceVec I)
    {q : ServiceVec I} {ε C : ℝ}
    (hq : Nonnegative q) (hε : 0 ≤ ε) (hscale : Nonnegative scale)
    (hscale_bound : ∀ i : I, scale i ≤ C)
    (hcoord : ∀ a ∈ F.acts, ∀ i : I,
      μtrue a i ≤ lower a i + ε * scale i) :
    support F μtrue q
      ≤ support F lower q + (ε * C) * l1 q := by
  have hsupport := coordinate_scaled_support_loss
    F μtrue lower scale hq hε hscale hcoord
  have hpressure := weightedPressure_le_const_l1
    (scale := scale) (q := q) hq hscale_bound
  have hscaled : ε * weightedPressure scale q ≤ ε * (C * l1 q) :=
    mul_le_mul_of_nonneg_left hpressure hε
  have hr : ε * (C * l1 q) = (ε * C) * l1 q := by ring
  linarith

/-! ## Coordinate-scaled lower confidence bounds -/

/-- Coordinate-wise lower-confidence service vector. -/
def coordinateLcbService
    (μhat rad : A → ServiceVec I) : A → ServiceVec I :=
  fun a i => μhat a i - rad a i

/-- Coordinate-wise confidence event. -/
def CoordinateServiceConfidenceEvent
    (F : ActionFamily A) (μtrue μhat rad : A → ServiceVec I) : Prop :=
  ∀ a ∈ F.acts, ∀ i : I, |μhat a i - μtrue a i| ≤ rad a i

/-- Coordinate confidence radii are dominated by a diagonal scale. -/
def CoordinateRadiusBoundedByScale
    (F : ActionFamily A) (rad : A → ServiceVec I)
    (scale : ServiceVec I) (ε : ℝ) : Prop :=
  ∀ a ∈ F.acts, ∀ i : I, 0 ≤ rad a i ∧ rad a i ≤ ε * scale i

/-- Coordinate LCB service is pessimistic on the confidence event. -/
theorem coordinateLcbService_le_true
    (F : ActionFamily A) (μtrue μhat rad : A → ServiceVec I)
    (hconf : CoordinateServiceConfidenceEvent F μtrue μhat rad) :
    ∀ a ∈ F.acts, ∀ i : I,
      coordinateLcbService μhat rad a i ≤ μtrue a i := by
  intro a ha i
  unfold coordinateLcbService
  have h := hconf a ha i
  have hle : μhat a i - μtrue a i ≤ rad a i := (abs_le.mp h).2
  linarith

/-- On the confidence event, true service is at most coordinate LCB service
plus two scaled radii. -/
theorem true_le_coordinateLcbService_add_two_scaled_radius
    (F : ActionFamily A) (μtrue μhat rad : A → ServiceVec I)
    (scale : ServiceVec I)
    {ε : ℝ}
    (hconf : CoordinateServiceConfidenceEvent F μtrue μhat rad)
    (hrad : CoordinateRadiusBoundedByScale F rad scale ε) :
    ∀ a ∈ F.acts, ∀ i : I,
      μtrue a i ≤ coordinateLcbService μhat rad a i + (2 * ε) * scale i := by
  intro a ha i
  unfold coordinateLcbService
  have h := hconf a ha i
  have hrad_ai := hrad a ha i
  have h1 : μtrue a i ≤ μhat a i + rad a i := by
    have hleft : -(rad a i) ≤ μhat a i - μtrue a i := (abs_le.mp h).1
    linarith
  have h2 : rad a i ≤ ε * scale i := hrad_ai.2
  linarith

/-- **Diagonal-scaled LCB support loss.**

The stochastic LCB loss is charged in normalized service units.  With
coordinate scales `scale_i`, replacing true service by coordinate LCB service
costs at most `2ε Σ_i scale_i Q_i`. -/
theorem true_support_le_coordinateLcb_support_plus_scaled_estimation_error
    (F : ActionFamily A) (μtrue μhat rad : A → ServiceVec I)
    (scale : ServiceVec I)
    {q : ServiceVec I} {ε : ℝ}
    (hq : Nonnegative q) (hε : 0 ≤ ε) (hscale : Nonnegative scale)
    (hconf : CoordinateServiceConfidenceEvent F μtrue μhat rad)
    (hrad : CoordinateRadiusBoundedByScale F rad scale ε) :
    support F μtrue q
      ≤ support F (coordinateLcbService μhat rad) q
          + (2 * ε) * weightedPressure scale q := by
  apply coordinate_scaled_support_loss
    F μtrue (coordinateLcbService μhat rad) scale hq (by nlinarith [hε])
      hscale
  intro a ha i
  exact true_le_coordinateLcbService_add_two_scaled_radius
    F μtrue μhat rad scale hconf hrad a ha i

/-- Diagonal-scaled LCB support loss converted to the ordinary drift norm. -/
theorem true_support_le_coordinateLcb_support_plus_scaled_estimation_error_l1
    (F : ActionFamily A) (μtrue μhat rad : A → ServiceVec I)
    (scale : ServiceVec I)
    {q : ServiceVec I} {ε C : ℝ}
    (hq : Nonnegative q) (hε : 0 ≤ ε) (hscale : Nonnegative scale)
    (hscale_bound : ∀ i : I, scale i ≤ C)
    (hconf : CoordinateServiceConfidenceEvent F μtrue μhat rad)
    (hrad : CoordinateRadiusBoundedByScale F rad scale ε) :
    support F μtrue q
      ≤ support F (coordinateLcbService μhat rad) q
          + ((2 * ε) * C) * l1 q := by
  apply coordinate_scaled_support_loss_l1
    F μtrue (coordinateLcbService μhat rad) scale hq
      (by nlinarith [hε]) hscale hscale_bound
  intro a ha i
  exact true_le_coordinateLcbService_add_two_scaled_radius
    F μtrue μhat rad scale hconf hrad a ha i

end Scheduleurm


/-! ## Source: Scheduleurm/RobustPolicy.lean -/


/-!
# Scheduleurm: robust candidate policy composition

This file connects the BAPR-HRO-style candidate re-ranking layer to the
MaxWeight drift layer.

The policy is selected from a candidate configuration set by maximizing

`qᵀ lower(a) - penalty(a)`,

where `lower(a)` may be an LCB / robust lower service vector and
`penalty(a)` collects switching, migration, rollback, OOM, bandwidth and
tail-risk terms.  The theorem composes four losses:

* candidate-set support loss;
* service-estimation / LCB support loss;
* re-ranking penalty;
* the remaining MaxWeight drift constant.
-/

noncomputable section

set_option linter.unusedSectionVars false
set_option linter.unusedVariables false

namespace Scheduleurm

variable {I A : Type*} [Fintype I] [DecidableEq A]

/-- A selected candidate maximizes robust score over the candidate set. -/
def RobustScoreMaximizer
    (cand : ActionFamily A) (lower : A → ServiceVec I)
    (q : ServiceVec I) (penalty : A → ℝ) (astar : A) : Prop :=
  astar ∈ cand.acts ∧
    ∀ b ∈ cand.acts,
      dot q (lower b) - penalty b ≤ dot q (lower astar) - penalty astar

/-- A selected candidate approximately maximizes robust score over the
candidate set, with additive score loss `α`. -/
def ApproxRobustScoreMaximizer
    (cand : ActionFamily A) (lower : A → ServiceVec I)
    (q : ServiceVec I) (penalty : A → ℝ) (astar : A) (α : ℝ) : Prop :=
  astar ∈ cand.acts ∧
    ∀ b ∈ cand.acts,
      dot q (lower b) - penalty b
        ≤ dot q (lower astar) - penalty astar + α

/-- Exact robust-score maximization is approximate maximization with zero
oracle loss. -/
theorem robustScoreMaximizer_is_approx
    (cand : ActionFamily A) (lower : A → ServiceVec I)
    (q : ServiceVec I) (penalty : A → ℝ) (astar : A)
    (hmax : RobustScoreMaximizer cand lower q penalty astar) :
    ApproxRobustScoreMaximizer cand lower q penalty astar 0 := by
  rcases hmax with ⟨hastar, hscore⟩
  refine ⟨hastar, ?_⟩
  intro b hb
  have h := hscore b hb
  linarith

/-- Penalties are nonnegative and uniformly bounded on the candidate set. -/
def PenaltyBounded
    (cand : ActionFamily A) (penalty : A → ℝ) (Pmax : ℝ) : Prop :=
  ∀ a ∈ cand.acts, 0 ≤ penalty a ∧ penalty a ≤ Pmax

/-- A robust-score maximizer is an approximate maximizer of candidate
lower-service support, with additive loss `Pmax`. -/
theorem robust_score_maximizer_support_bound
    (cand : ActionFamily A) (lower : A → ServiceVec I)
    {q : ServiceVec I} (penalty : A → ℝ) {astar : A} {Pmax : ℝ}
    (hmax : RobustScoreMaximizer cand lower q penalty astar)
    (hpen : PenaltyBounded cand penalty Pmax) :
    support cand lower q ≤ dot q (lower astar) + Pmax := by
  rcases hmax with ⟨hastar, hscore⟩
  apply support_le
  intro b hb
  have hs := hscore b hb
  have hpen_b := hpen b hb
  have hpen_a := hpen astar hastar
  linarith

/-- An approximate robust-score maximizer is an approximate maximizer of
candidate lower-service support, with additive loss `Pmax + α`. -/
theorem approx_robust_score_maximizer_support_bound
    (cand : ActionFamily A) (lower : A → ServiceVec I)
    {q : ServiceVec I} (penalty : A → ℝ) {astar : A} {Pmax α : ℝ}
    (hmax : ApproxRobustScoreMaximizer cand lower q penalty astar α)
    (hpen : PenaltyBounded cand penalty Pmax) :
    support cand lower q ≤ dot q (lower astar) + Pmax + α := by
  rcases hmax with ⟨hastar, hscore⟩
  apply support_le
  intro b hb
  have hs := hscore b hb
  have hpen_b := hpen b hb
  have hpen_a := hpen astar hastar
  linarith

/-- **Robust candidate re-ranking support theorem.**

If
* the candidate set approximates the full true-service support by `εcand`;
* the lower-service model approximates candidate true support by `εest`;
* the selected action maximizes the lower-service score up to bounded
  penalty `Pmax`;

then the selected action is an approximate full MaxWeight action with total
loss `(εcand + εest)||q||₁ + Pmax`. -/
theorem robust_candidate_policy_approx_full_support
    (full cand : ActionFamily A)
    (μtrue lower : A → ServiceVec I)
    {q : ServiceVec I} {εcand εest Pmax : ℝ}
    (hq : Nonnegative q)
    (hgap : SupportGapAtMost full cand μtrue εcand)
    (hlower_gap : support cand μtrue q
        ≤ support cand lower q + εest * l1 q)
    (penalty : A → ℝ) {astar : A}
    (hmax : RobustScoreMaximizer cand lower q penalty astar)
    (hpen : PenaltyBounded cand penalty Pmax) :
    support full μtrue q
      ≤ dot q (lower astar) + (εcand + εest) * l1 q + Pmax := by
  have h1 : support full μtrue q
      ≤ support cand μtrue q + εcand * l1 q := hgap q hq
  have h2 : support full μtrue q
      ≤ support cand lower q + εest * l1 q + εcand * l1 q := by
    linarith
  have h3 : support cand lower q ≤ dot q (lower astar) + Pmax :=
    robust_score_maximizer_support_bound cand lower penalty hmax hpen
  nlinarith

/-- The robust candidate policy can be fed directly into the approximate
MaxWeight drift theorem. -/
theorem robust_candidate_policy_lyapunov_drift
    (full cand : ActionFamily A)
    (μtrue lower : A → ServiceVec I)
    {Q lam : ServiceVec I} {δ εcand εest B Pmax : ℝ}
    (hQ : Nonnegative Q) (hlam : Nonnegative lam)
    (hcap : InCapacityWithSlack full μtrue lam δ)
    (hgap : SupportGapAtMost full cand μtrue εcand)
    (hlower_gap : support cand μtrue Q
        ≤ support cand lower Q + εest * l1 Q)
    (penalty : A → ℝ) {astar : A}
    (hmax : RobustScoreMaximizer cand lower Q penalty astar)
    (hpen : PenaltyBounded cand penalty Pmax)
    (hSecond : secondOrderTerm lam (lower astar) ≤ B)
    (hlower_a_nonneg : Nonnegative (lower astar)) :
    Lyapunov (queueStep Q lam (lower astar)) - Lyapunov Q
      ≤ B + Pmax - (δ - (εcand + εest)) * l1 Q := by
  have hApprox : support full μtrue Q
      ≤ dot Q (lower astar) + (εcand + εest) * l1 Q + Pmax :=
    robust_candidate_policy_approx_full_support full cand μtrue lower hQ hgap
      hlower_gap penalty hmax hpen
  have hApprox' : support full μtrue Q
      ≤ dot Q (lower astar) + (εcand + εest) * l1 Q + Pmax := hApprox
  exact approximate_maxWeight_lyapunov_drift full μtrue hQ hlam hlower_a_nonneg
    hcap hApprox' hSecond

end Scheduleurm


/-! ## Source: Scheduleurm/PenaltyGrowth.lean -/


/-!
# Scheduleurm: bounded and queue-growing penalties

Reviewers will object if switching/risk penalties in

`Qᵀ μ(a) - K(a_prev,a) - R(a)`

are allowed to grow with backlog without being charged against capacity
slack.  This file makes the boundary explicit.  Uniformly bounded penalties
cost an additive constant; penalties that grow at rate `β ||Q||₁` reduce the
MaxWeight stability margin by `β`.
-/

noncomputable section

set_option linter.unusedSectionVars false
set_option linter.unusedVariables false

namespace Scheduleurm

variable {I A : Type*} [Fintype I] [DecidableEq A]

/-- Penalties may have a bounded part and a backlog-proportional part. -/
def QueueScaledPenaltyBounded
    (cand : ActionFamily A) (penalty : ServiceVec I → A → ℝ)
    (P0 β : ℝ) : Prop :=
  ∀ Q : ServiceVec I, ∀ a ∈ cand.acts,
    0 ≤ penalty Q a ∧ penalty Q a ≤ P0 + β * l1 Q

/-- Queue-scaled approximate robust-score maximization.  This matches
practical solvers such as greedy search, local search, or time-limited ILP:
the selected score may be below the candidate optimum by a bounded part
`α0` and a backlog-proportional part `α1 ||Q||₁`. -/
def QueueScaledApproxRobustScoreMaximizer
    (cand : ActionFamily A) (lower : A → ServiceVec I)
    (q : ServiceVec I) (penalty : ServiceVec I → A → ℝ)
    (astar : A) (α0 α1 : ℝ) : Prop :=
  ApproxRobustScoreMaximizer cand lower q (penalty q) astar
    (α0 + α1 * l1 q)

/-- Queue-scaled penalties turn a robust-score maximizer into an approximate
candidate support maximizer with loss `P0 + β ||Q||₁`. -/
theorem robust_score_maximizer_support_bound_scaled
    (cand : ActionFamily A) (lower : A → ServiceVec I)
    {q : ServiceVec I} (penalty : ServiceVec I → A → ℝ)
    {astar : A} {P0 β : ℝ}
    (hmax : RobustScoreMaximizer cand lower q (penalty q) astar)
    (hpen : QueueScaledPenaltyBounded cand penalty P0 β) :
    support cand lower q ≤ dot q (lower astar) + P0 + β * l1 q := by
  rcases hmax with ⟨hastar, hscore⟩
  apply support_le
  intro b hb
  have hs := hscore b hb
  have hpen_b := hpen q b hb
  have hpen_a := hpen q astar hastar
  linarith

/-- Queue-scaled approximate solvers consume additive constant `α0` and
capacity slack `α1`. -/
theorem approx_robust_score_maximizer_support_bound_scaled
    (cand : ActionFamily A) (lower : A → ServiceVec I)
    {q : ServiceVec I} (penalty : ServiceVec I → A → ℝ)
    {astar : A} {P0 β α0 α1 : ℝ}
    (hmax :
      QueueScaledApproxRobustScoreMaximizer cand lower q penalty astar α0 α1)
    (hpen : QueueScaledPenaltyBounded cand penalty P0 β) :
    support cand lower q
      ≤ dot q (lower astar) + (P0 + α0) + (β + α1) * l1 q := by
  unfold QueueScaledApproxRobustScoreMaximizer at hmax
  have h :=
    approx_robust_score_maximizer_support_bound
      cand lower (penalty q) hmax
      (Pmax := P0 + β * l1 q) (α := α0 + α1 * l1 q)
      (by
        intro a ha
        exact hpen q a ha)
  linarith

/-- Queue-growing penalties consume capacity slack exactly like candidate and
estimation errors. -/
theorem robust_candidate_policy_approx_full_support_scaled_penalty
    (full cand : ActionFamily A)
    (μtrue lower : A → ServiceVec I)
    {q : ServiceVec I} {εcand εest P0 β : ℝ}
    (hq : Nonnegative q)
    (hgap : SupportGapAtMost full cand μtrue εcand)
    (hlower_gap : support cand μtrue q
        ≤ support cand lower q + εest * l1 q)
    (penalty : ServiceVec I → A → ℝ) {astar : A}
    (hmax : RobustScoreMaximizer cand lower q (penalty q) astar)
    (hpen : QueueScaledPenaltyBounded cand penalty P0 β) :
    support full μtrue q
      ≤ dot q (lower astar) + (εcand + εest + β) * l1 q + P0 := by
  have h1 : support full μtrue q
      ≤ support cand μtrue q + εcand * l1 q := hgap q hq
  have h2 : support full μtrue q
      ≤ support cand lower q + εest * l1 q + εcand * l1 q := by
    linarith
  have h3 : support cand lower q
      ≤ dot q (lower astar) + P0 + β * l1 q :=
    robust_score_maximizer_support_bound_scaled cand lower penalty hmax hpen
  nlinarith

/-- Candidate approximation, estimation error, queue-scaled penalty, and
approximate optimization oracle error combine additively in the full-support
loss. -/
theorem robust_candidate_policy_approx_full_support_scaled_penalty_approx_oracle
    (full cand : ActionFamily A)
    (μtrue lower : A → ServiceVec I)
    {q : ServiceVec I} {εcand εest P0 β α0 α1 : ℝ}
    (hq : Nonnegative q)
    (hgap : SupportGapAtMost full cand μtrue εcand)
    (hlower_gap : support cand μtrue q
        ≤ support cand lower q + εest * l1 q)
    (penalty : ServiceVec I → A → ℝ) {astar : A}
    (hmax :
      QueueScaledApproxRobustScoreMaximizer cand lower q penalty astar α0 α1)
    (hpen : QueueScaledPenaltyBounded cand penalty P0 β) :
    support full μtrue q
      ≤ dot q (lower astar) + (εcand + εest + β + α1) * l1 q
          + (P0 + α0) := by
  have h1 : support full μtrue q
      ≤ support cand μtrue q + εcand * l1 q := hgap q hq
  have h2 : support full μtrue q
      ≤ support cand lower q + εest * l1 q + εcand * l1 q := by
    linarith
  have h3 : support cand lower q
      ≤ dot q (lower astar) + (P0 + α0) + (β + α1) * l1 q :=
    approx_robust_score_maximizer_support_bound_scaled
      cand lower penalty hmax hpen
  nlinarith

/-- Drift bound when penalties grow at rate `β ||Q||₁`.  Stability now
requires `δ > εcand + εest + β`. -/
theorem robust_candidate_policy_lyapunov_drift_scaled_penalty
    (full cand : ActionFamily A)
    (μtrue lower : A → ServiceVec I)
    {Q lam : ServiceVec I} {δ εcand εest B P0 β : ℝ}
    (hQ : Nonnegative Q) (hlam : Nonnegative lam)
    (hcap : InCapacityWithSlack full μtrue lam δ)
    (hgap : SupportGapAtMost full cand μtrue εcand)
    (hlower_gap : support cand μtrue Q
        ≤ support cand lower Q + εest * l1 Q)
    (penalty : ServiceVec I → A → ℝ) {astar : A}
    (hmax : RobustScoreMaximizer cand lower Q (penalty Q) astar)
    (hpen : QueueScaledPenaltyBounded cand penalty P0 β)
    (hSecond : secondOrderTerm lam (lower astar) ≤ B)
    (hlower_a_nonneg : Nonnegative (lower astar)) :
    Lyapunov (queueStep Q lam (lower astar)) - Lyapunov Q
      ≤ B + P0 - (δ - (εcand + εest + β)) * l1 Q := by
  have hApprox : support full μtrue Q
      ≤ dot Q (lower astar) + (εcand + εest + β) * l1 Q + P0 :=
    robust_candidate_policy_approx_full_support_scaled_penalty
      full cand μtrue lower hQ hgap hlower_gap penalty hmax hpen
  exact approximate_maxWeight_lyapunov_drift full μtrue hQ hlam hlower_a_nonneg
    hcap hApprox hSecond

/-- Drift bound for a queue-scaled approximate robust-score oracle.  The
oracle's backlog-proportional loss `α1` consumes slack, while `α0` adds to
the constant term. -/
theorem robust_candidate_policy_lyapunov_drift_scaled_penalty_approx_oracle
    (full cand : ActionFamily A)
    (μtrue lower : A → ServiceVec I)
    {Q lam : ServiceVec I} {δ εcand εest B P0 β α0 α1 : ℝ}
    (hQ : Nonnegative Q) (hlam : Nonnegative lam)
    (hcap : InCapacityWithSlack full μtrue lam δ)
    (hgap : SupportGapAtMost full cand μtrue εcand)
    (hlower_gap : support cand μtrue Q
        ≤ support cand lower Q + εest * l1 Q)
    (penalty : ServiceVec I → A → ℝ) {astar : A}
    (hmax :
      QueueScaledApproxRobustScoreMaximizer cand lower Q penalty astar α0 α1)
    (hpen : QueueScaledPenaltyBounded cand penalty P0 β)
    (hSecond : secondOrderTerm lam (lower astar) ≤ B)
    (hlower_a_nonneg : Nonnegative (lower astar)) :
    Lyapunov (queueStep Q lam (lower astar)) - Lyapunov Q
      ≤ B + (P0 + α0)
          - (δ - (εcand + εest + β + α1)) * l1 Q := by
  have hApprox : support full μtrue Q
      ≤ dot Q (lower astar) + (εcand + εest + β + α1) * l1 Q
          + (P0 + α0) :=
    robust_candidate_policy_approx_full_support_scaled_penalty_approx_oracle
      full cand μtrue lower hQ hgap hlower_gap penalty hmax hpen
  exact approximate_maxWeight_lyapunov_drift full μtrue hQ hlam hlower_a_nonneg
    hcap hApprox hSecond

end Scheduleurm


/-! ## Source: Scheduleurm/PiecewiseStationary.lean -/


/-!
# Scheduleurm: piecewise-stationary hidden-regime penalties

This file formalises the finite-horizon accounting layer behind Theorem C
in `math.md`.

The hidden regime may be stationary only between unknown change points.  The
queueing-control proof then needs a deterministic bridge:

* normal slots contribute the usual negative MaxWeight drift;
* change-point detection windows contribute a delay/backlog penalty;
* service-learning errors contribute an estimation penalty;
* switching/migration costs remain explicit.

The theorems below are deliberately stated as finite-sum inequalities.  This
keeps the stochastic assumptions separate from the queueing-control algebra:
once a detector supplies per-change bounds, they can be plugged into the
cumulative Lyapunov drift argument without changing the scheduler proof.
-/

noncomputable section

set_option linter.unusedSectionVars false
set_option linter.unusedVariables false

namespace Scheduleurm

open BigOperators

variable {I T K Z : Type*} [Fintype I] [Fintype T] [Fintype K]

/-- A finite-horizon regime path is piecewise constant with respect to a
segment label if equal segment labels imply equal regimes. -/
def PiecewiseConstantOn (segmentOf : T → K) (regime : T → Z) : Prop :=
  ∀ t u : T, segmentOf t = segmentOf u → regime t = regime u

/-- Detection-window cost `Σ_k delay_k * backlog_k`.  This is the formal
version of `Σ τ_detect,k |Q(τ_k)|` in Theorem C. -/
def detectionPenalty (delay backlogAtChange : K → ℝ) : ℝ :=
  ∑ k : K, delay k * backlogAtChange k

/-- Sum of per-slot estimation losses. -/
def estimationPenalty (estimationLoss : T → ℝ) : ℝ :=
  ∑ t : T, estimationLoss t

/-- Per-change detection losses are bounded by delay times backlog, hence so
is their cumulative detection-window contribution. -/
theorem detection_window_penalty_bound
    (changeLoss delay backlogAtChange : K → ℝ)
    (hlocal : ∀ k : K, changeLoss k ≤ delay k * backlogAtChange k) :
    (∑ k : K, changeLoss k) ≤ detectionPenalty delay backlogAtChange := by
  unfold detectionPenalty
  exact Finset.sum_le_sum (fun k _ => hlocal k)

/-- **Piecewise-stationary penalty accounting.**

If the non-stationarity overhead is bounded by the sum of local change-point
losses, estimation losses, and switching/migration cost, and each local
change-point loss is bounded by `delay_k * |Q(τ_k)|`, then the total overhead
is bounded by the Theorem-C expression.
-/
theorem piecewise_stationary_penalty_bound
    (changeLoss delay backlogAtChange : K → ℝ)
    (estimationLoss : T → ℝ) {switchingCost totalOverhead : ℝ}
    (htotal : totalOverhead
      ≤ (∑ k : K, changeLoss k) + estimationPenalty estimationLoss
          + switchingCost)
    (hlocal : ∀ k : K, changeLoss k ≤ delay k * backlogAtChange k) :
    totalOverhead
      ≤ detectionPenalty delay backlogAtChange
          + estimationPenalty estimationLoss + switchingCost := by
  have hdetect :
      (∑ k : K, changeLoss k) ≤ detectionPenalty delay backlogAtChange :=
    detection_window_penalty_bound changeLoss delay backlogAtChange hlocal
  linarith

/-- Expands the normal-slot cumulative drift term into the standard
`T·B - margin·Σ||Q_t||₁ + estimation` form. -/
lemma sum_drift_template_eq
    (Q : T → ServiceVec I) (estimationLoss : T → ℝ)
    (B margin : ℝ) :
    (∑ t : T, (B - margin * l1 (Q t) + estimationLoss t))
      =
    (Fintype.card T : ℝ) * B
      - margin * (∑ t : T, l1 (Q t))
      + ∑ t : T, estimationLoss t := by
  rw [Finset.sum_add_distrib, Finset.sum_sub_distrib]
  rw [Finset.mul_sum]
  simp

/-- **Cumulative piecewise-stationary drift bound.**

This is the finite-horizon Lyapunov accounting statement used after the
single-step robust MaxWeight drift theorem.  The normal slots provide a
negative backlog term with margin `margin`; change-point windows and
estimation errors are carried explicitly.
-/
theorem cumulative_drift_with_piecewise_detection
    (drift : T → ℝ) (Q : T → ServiceVec I)
    (changeLoss delay backlogAtChange : K → ℝ)
    (estimationLoss : T → ℝ) {B margin switchingCost : ℝ}
    (hstep : ∀ t : T,
      drift t ≤ B - margin * l1 (Q t) + estimationLoss t)
    (hlocal : ∀ k : K, changeLoss k ≤ delay k * backlogAtChange k) :
    (∑ t : T, drift t) + (∑ k : K, changeLoss k) + switchingCost
      ≤
    (Fintype.card T : ℝ) * B
      - margin * (∑ t : T, l1 (Q t))
      + estimationPenalty estimationLoss
      + detectionPenalty delay backlogAtChange
      + switchingCost := by
  have hstep_sum :
      (∑ t : T, drift t)
        ≤ ∑ t : T, (B - margin * l1 (Q t) + estimationLoss t) := by
    exact Finset.sum_le_sum (fun t _ => hstep t)
  have hdetect :
      (∑ k : K, changeLoss k) ≤ detectionPenalty delay backlogAtChange :=
    detection_window_penalty_bound changeLoss delay backlogAtChange hlocal
  have htemplate := sum_drift_template_eq Q estimationLoss B margin
  unfold estimationPenalty
  linarith

/-! ## Dwell-time / switching-window backlog budgets -/

/-- Backlog mass inside one regime segment. -/
def segmentBacklog
    [DecidableEq K]
    (segmentOf : T → K) (Q : T → ServiceVec I) (k : K) : ℝ :=
  ∑ t ∈ (Finset.univ : Finset T) with segmentOf t = k, l1 (Q t)

/-- Backlog mass over marked slots, e.g. detection windows or forced
switching/migration windows. -/
def markedBacklog
    (marked : T → Prop) [DecidablePred marked]
    (Q : T → ServiceVec I) : ℝ :=
  ∑ t ∈ (Finset.univ.filter marked), l1 (Q t)

/-- Marked backlog mass inside one segment. -/
def markedBacklogBySegment
    [DecidableEq K]
    (segmentOf : T → K) (marked : T → Prop) [DecidablePred marked]
    (Q : T → ServiceVec I) (k : K) : ℝ :=
  ∑ t ∈ (Finset.univ.filter marked) with segmentOf t = k, l1 (Q t)

/-- Segment backlog decomposes total backlog. -/
lemma sum_segmentBacklog_eq_total
    [DecidableEq K] (segmentOf : T → K) (Q : T → ServiceVec I) :
    (∑ k : K, segmentBacklog segmentOf Q k) = ∑ t : T, l1 (Q t) := by
  unfold segmentBacklog
  simpa using
    (Finset.sum_fiberwise (Finset.univ : Finset T) segmentOf
      (fun t => l1 (Q t)))

/-- Marked backlog decomposes over segments. -/
lemma sum_markedBacklogBySegment_eq_markedBacklog
    [DecidableEq K] (segmentOf : T → K)
    (marked : T → Prop) [DecidablePred marked]
    (Q : T → ServiceVec I) :
    (∑ k : K, markedBacklogBySegment segmentOf marked Q k)
      = markedBacklog marked Q := by
  unfold markedBacklogBySegment markedBacklog
  simpa using
    (Finset.sum_fiberwise (Finset.univ.filter marked) segmentOf
      (fun t => l1 (Q t)))

/-- If every segment spends at most a `θ` fraction of its backlog mass inside
detection/switching windows, up to a fixed residual, then the total marked
backlog obeys the same global budget.  This is the finite-horizon form of a
dwell-time condition. -/
theorem marked_backlog_le_fraction_total_from_segments
    [DecidableEq K]
    (segmentOf : T → K) (marked : T → Prop) [DecidablePred marked]
    (Q : T → ServiceVec I) (fixed : K → ℝ) {θ : ℝ}
    (hseg : ∀ k : K,
      markedBacklogBySegment segmentOf marked Q k
        ≤ θ * segmentBacklog segmentOf Q k + fixed k) :
    markedBacklog marked Q
      ≤ θ * (∑ t : T, l1 (Q t)) + ∑ k : K, fixed k := by
  have hsum :
      (∑ k : K, markedBacklogBySegment segmentOf marked Q k)
        ≤ ∑ k : K, (θ * segmentBacklog segmentOf Q k + fixed k) := by
    exact Finset.sum_le_sum (fun k _ => hseg k)
  have hmarked :=
    sum_markedBacklogBySegment_eq_markedBacklog segmentOf marked Q
  have hsegment := sum_segmentBacklog_eq_total segmentOf Q
  rw [Finset.sum_add_distrib] at hsum
  rw [← Finset.mul_sum] at hsum
  rw [hmarked, hsegment] at hsum
  exact hsum

/-- Cumulative drift with dwell-time/switching-window accounting.

Marked slots may add backlog-proportional overhead at rate `χ`.  If the
marked backlog is at most a `θ` fraction of total backlog in each segment,
then the negative drift margin is reduced from `margin` to
`margin - χθ`, plus only a fixed per-segment residual. -/
theorem cumulative_drift_with_dwell_switching_budget
    [DecidableEq K]
    (drift : T → ℝ) (Q : T → ServiceVec I)
    (segmentOf : T → K) (marked : T → Prop) [DecidablePred marked]
    (fixed : K → ℝ) (estimationLoss : T → ℝ)
    {B margin χ θ switchingCost : ℝ}
    (hχ : 0 ≤ χ)
    (hstep : ∀ t : T,
      drift t ≤ B - margin * l1 (Q t) + estimationLoss t)
    (hseg : ∀ k : K,
      markedBacklogBySegment segmentOf marked Q k
        ≤ θ * segmentBacklog segmentOf Q k + fixed k) :
    (∑ t : T, drift t) + χ * markedBacklog marked Q + switchingCost
      ≤
    (Fintype.card T : ℝ) * B
      - (margin - χ * θ) * (∑ t : T, l1 (Q t))
      + estimationPenalty estimationLoss
      + χ * (∑ k : K, fixed k)
      + switchingCost := by
  have hstep_sum :
      (∑ t : T, drift t)
        ≤ ∑ t : T, (B - margin * l1 (Q t) + estimationLoss t) := by
    exact Finset.sum_le_sum (fun t _ => hstep t)
  have htemplate := sum_drift_template_eq Q estimationLoss B margin
  have hmarked := marked_backlog_le_fraction_total_from_segments
    segmentOf marked Q fixed hseg
  have hmarked_mul :
      χ * markedBacklog marked Q
        ≤ χ * (θ * (∑ t : T, l1 (Q t)) + ∑ k : K, fixed k) :=
    mul_le_mul_of_nonneg_left hmarked hχ
  unfold estimationPenalty
  nlinarith

end Scheduleurm


/-! ## Source: Scheduleurm/Regret.lean -/


/-!
# Scheduleurm: cumulative support regret and learning bounds

This file formalises the finite-horizon regret layer from Theorem D in
`math.md`.

The statements keep the two relevant notions separate:

* **support regret**: the queue-weighted service gap relative to the full
  true MaxWeight oracle;
* **learning regret**: the extra support loss incurred by replacing true
  service with lower-confidence service.

These cumulative inequalities are the pieces needed before adding a
statistical concentration theorem for the usual
`~sqrt(|A||Z|T)` confidence-radius sum.
-/

noncomputable section

set_option linter.unusedSectionVars false
set_option linter.unusedVariables false

namespace Scheduleurm

open BigOperators

variable {I A T : Type*} [Fintype I] [DecidableEq A] [Fintype T]

/-- Finite-horizon regret between an oracle score and a chosen score. -/
def cumulativeRegret (oracle chosen : T → ℝ) : ℝ :=
  ∑ t : T, (oracle t - chosen t)

/-- Per-slot regret upper bounds sum to a cumulative regret upper bound. -/
theorem cumulative_regret_from_step_bounds
    (oracle chosen bound : T → ℝ)
    (hstep : ∀ t : T, oracle t - chosen t ≤ bound t) :
    cumulativeRegret oracle chosen ≤ ∑ t : T, bound t := by
  unfold cumulativeRegret
  exact Finset.sum_le_sum (fun t _ => hstep t)

/-- **Cumulative robust-candidate support regret.**

This is the finite-horizon version of the robust candidate policy theorem:
the full true MaxWeight support gap is bounded by the sum of candidate-cover
losses, estimation/model losses, and robust re-ranking penalties.
-/
theorem cumulative_robust_candidate_support_regret
    (full cand : ActionFamily A)
    (μtrue lower : A → ServiceVec I)
    (Q : T → ServiceVec I)
    (εcand εest Pmax : T → ℝ)
    (penalty : T → A → ℝ) (astar : T → A)
    (hQ : ∀ t : T, Nonnegative (Q t))
    (hgap : ∀ t : T, SupportGapAtMost full cand μtrue (εcand t))
    (hlower_gap : ∀ t : T,
      support cand μtrue (Q t)
        ≤ support cand lower (Q t) + εest t * l1 (Q t))
    (hmax : ∀ t : T,
      RobustScoreMaximizer cand lower (Q t) (penalty t) (astar t))
    (hpen : ∀ t : T, PenaltyBounded cand (penalty t) (Pmax t)) :
    (∑ t : T,
      (support full μtrue (Q t) - dot (Q t) (lower (astar t))))
      ≤
    ∑ t : T, ((εcand t + εest t) * l1 (Q t) + Pmax t) := by
  apply Finset.sum_le_sum
  intro t _
  have h := robust_candidate_policy_approx_full_support
    full cand μtrue lower (hQ t) (hgap t) (hlower_gap t)
    (penalty t) (hmax t) (hpen t)
  linarith

/-- **Cumulative LCB learning support regret.**

Under the confidence event at every slot, replacing true service by
lower-confidence service costs at most `2ε_t ||Q_t||₁` in support value, and
therefore the cumulative learning regret is bounded by the corresponding
finite sum.
-/
theorem cumulative_lcb_learning_support_regret
    (F : ActionFamily A)
    (μtrue μhat : T → A → ServiceVec I)
    (rad : T → A → ℝ)
    (Q : T → ServiceVec I) (ε : T → ℝ)
    (hQ : ∀ t : T, Nonnegative (Q t))
    (hε : ∀ t : T, 0 ≤ ε t)
    (hconf : ∀ t : T,
      ServiceConfidenceEvent F (μtrue t) (μhat t) (rad t))
    (hrad : ∀ t : T, RadiusBounded F (rad t) (ε t)) :
    (∑ t : T,
      (support F (μtrue t) (Q t)
        - support F (lcbService (μhat t) (rad t)) (Q t)))
      ≤
    ∑ t : T, ((2 * ε t) * l1 (Q t)) := by
  apply Finset.sum_le_sum
  intro t _
  have h := true_support_le_lcb_support_plus_estimation_error
    F (μtrue t) (μhat t) (rad t) (hQ t) (hε t) (hconf t) (hrad t)
  linarith

/-- A uniform per-slot loss bound yields a linear finite-horizon bound.  This
lemma is useful when the statistical analysis has already reduced the
confidence-radius term to a deterministic envelope. -/
theorem cumulative_loss_under_uniform_bound
    (loss : T → ℝ) {C : ℝ}
    (hstep : ∀ t : T, loss t ≤ C) :
    (∑ t : T, loss t) ≤ (Fintype.card T : ℝ) * C := by
  calc
    (∑ t : T, loss t) ≤ ∑ _t : T, C := by
      exact Finset.sum_le_sum (fun t _ => hstep t)
    _ = (Fintype.card T : ℝ) * C := by
      simp

/-- **Learning + change-point + switching regret decomposition.**

This is the deterministic shell of the Theorem-D
`learning + N_cp log T + switching` bound.  A concentration theorem can
instantiate `statisticalEnvelope`; a detector analysis can instantiate
`changeEnvelope`.
-/
theorem learning_change_switching_regret_decomposition
    {totalRegret learningLoss changePointLoss switchingCost
      statisticalEnvelope changeEnvelope : ℝ}
    (htotal : totalRegret ≤ learningLoss + changePointLoss + switchingCost)
    (hlearn : learningLoss ≤ statisticalEnvelope)
    (hchange : changePointLoss ≤ changeEnvelope) :
    totalRegret ≤ statisticalEnvelope + changeEnvelope + switchingCost := by
  linarith

end Scheduleurm


/-! ## Source: Scheduleurm/FosterLyapunov.lean -/


/-!
# Scheduleurm: Foster-Lyapunov stability certificates

This file adds the outer stability layer behind the robust MaxWeight drift
theorem.

Mathlib does not currently provide a ready-made positive-recurrence theorem
for countable-state Markov chains.  We therefore formalise the queueing part
that such a theorem consumes:

* a Foster negative-drift certificate outside large backlogs;
* the finite-horizon telescoping consequence bounding cumulative backlog;
* a theorem showing that the robust candidate MaxWeight policy supplies the
  required negative-drift certificate whenever its total approximation loss is
  smaller than the capacity slack.

This is the exact algebraic core of the standard Foster-Lyapunov proof of
positive recurrence; the remaining measure-theoretic step can be attached
later if a Markov-chain recurrence library is introduced.
-/

noncomputable section

set_option linter.unusedSectionVars false
set_option linter.unusedVariables false

namespace Scheduleurm

open BigOperators

variable {I A T : Type*} [Fintype I] [DecidableEq A] [Fintype T]

/-- Foster negative drift with respect to `||Q||₁`. -/
def FosterNegativeDrift
    (drift : ServiceVec I → ℝ) (B η : ℝ) : Prop :=
  ∀ Q : ServiceVec I, Nonnegative Q → drift Q ≤ B - η * l1 Q

/-- A positive-recurrence-ready Foster certificate: positive drift margin and
linear negative Lyapunov drift outside bounded sets. -/
def FosterPositiveRecurrenceCertificate
    (drift : ServiceVec I → ℝ) : Prop :=
  ∃ B η : ℝ, 0 < η ∧ FosterNegativeDrift drift B η

/-- The usual finite-horizon telescoping consequence of Foster negative
drift.  If the sum of one-step drifts telescopes to `V_T - V_0`, and
`V_T ≥ 0`, then cumulative backlog is bounded by `(T B + V_0)/η`. -/
theorem foster_telescoping_l1_bound
    (drift : T → ℝ) (Q : T → ServiceVec I)
    {B η V0 VT : ℝ}
    (hη : 0 < η)
    (hstep : ∀ t : T, drift t ≤ B - η * l1 (Q t))
    (htelescope : (∑ t : T, drift t) = VT - V0)
    (hVT : 0 ≤ VT) :
    η * (∑ t : T, l1 (Q t)) ≤ (Fintype.card T : ℝ) * B + V0 := by
  have hsum :
      (∑ t : T, drift t)
        ≤ ∑ t : T, (B - η * l1 (Q t)) := by
    exact Finset.sum_le_sum (fun t _ => hstep t)
  have htemplate :
      (∑ t : T, (B - η * l1 (Q t)))
        = (Fintype.card T : ℝ) * B - η * (∑ t : T, l1 (Q t)) := by
    rw [Finset.sum_sub_distrib, Finset.mul_sum]
    simp
  linarith

/-- Robust candidate MaxWeight supplies a Foster negative-drift certificate
when the candidate, estimation and robust-score losses leave positive slack. -/
theorem robust_candidate_policy_foster_negative_drift
    (full cand : ActionFamily A)
    (μtrue lower : A → ServiceVec I)
    (lam : ServiceVec I) {δ εcand εest B Pmax η : ℝ}
    (hlam : Nonnegative lam)
    (hcap : InCapacityWithSlack full μtrue lam δ)
    (hgap : SupportGapAtMost full cand μtrue εcand)
    (penalty : ServiceVec I → A → ℝ)
    (astar : ServiceVec I → A)
    (hlower_gap : ∀ Q : ServiceVec I, Nonnegative Q →
      support cand μtrue Q ≤ support cand lower Q + εest * l1 Q)
    (hmax : ∀ Q : ServiceVec I, Nonnegative Q →
      RobustScoreMaximizer cand lower Q (penalty Q) (astar Q))
    (hpen : ∀ Q : ServiceVec I, Nonnegative Q →
      PenaltyBounded cand (penalty Q) Pmax)
    (hSecond : ∀ Q : ServiceVec I, Nonnegative Q →
      secondOrderTerm lam (lower (astar Q)) ≤ B)
    (hlower_nonneg : ∀ Q : ServiceVec I, Nonnegative Q →
      Nonnegative (lower (astar Q)))
    (hη : η = δ - (εcand + εest)) :
    FosterNegativeDrift
      (fun Q =>
        Lyapunov (queueStep Q lam (lower (astar Q))) - Lyapunov Q)
      (B + Pmax) η := by
  intro Q hQ
  have hdrift := robust_candidate_policy_lyapunov_drift
    full cand μtrue lower hQ hlam hcap hgap (hlower_gap Q hQ)
    (penalty Q) (hmax Q hQ) (hpen Q hQ) (hSecond Q hQ)
    (hlower_nonneg Q hQ)
  subst η
  exact hdrift

/-- Positive-margin robust candidate MaxWeight yields a
positive-recurrence-ready Foster certificate. -/
theorem robust_candidate_policy_positive_recurrence_certificate
    (full cand : ActionFamily A)
    (μtrue lower : A → ServiceVec I)
    (lam : ServiceVec I) {δ εcand εest B Pmax : ℝ}
    (hlam : Nonnegative lam)
    (hcap : InCapacityWithSlack full μtrue lam δ)
    (hgap : SupportGapAtMost full cand μtrue εcand)
    (penalty : ServiceVec I → A → ℝ)
    (astar : ServiceVec I → A)
    (hlower_gap : ∀ Q : ServiceVec I, Nonnegative Q →
      support cand μtrue Q ≤ support cand lower Q + εest * l1 Q)
    (hmax : ∀ Q : ServiceVec I, Nonnegative Q →
      RobustScoreMaximizer cand lower Q (penalty Q) (astar Q))
    (hpen : ∀ Q : ServiceVec I, Nonnegative Q →
      PenaltyBounded cand (penalty Q) Pmax)
    (hSecond : ∀ Q : ServiceVec I, Nonnegative Q →
      secondOrderTerm lam (lower (astar Q)) ≤ B)
    (hlower_nonneg : ∀ Q : ServiceVec I, Nonnegative Q →
      Nonnegative (lower (astar Q)))
    (hmargin : 0 < δ - (εcand + εest)) :
    FosterPositiveRecurrenceCertificate
      (fun Q =>
        Lyapunov (queueStep Q lam (lower (astar Q))) - Lyapunov Q) := by
  refine ⟨B + Pmax, δ - (εcand + εest), hmargin, ?_⟩
  exact robust_candidate_policy_foster_negative_drift
    full cand μtrue lower lam hlam hcap hgap penalty astar
    hlower_gap hmax hpen hSecond hlower_nonneg rfl

end Scheduleurm


/-! ## Source: Scheduleurm/MarkovRecurrence.lean -/


/-!
# Scheduleurm: a minimal Markov-chain recurrence layer

This file provides the recurrence-theorem fragment needed by the
configuration scheduler proof.

Mathlib has Markov kernels, martingales and optional sampling, but not a
ready-made countable-state positive-recurrence API.  Instead of inventing a
fake `PositiveRecurrent` theorem, we isolate the exact Foster-Lyapunov step
that is both standard and needed here:

If a Markov transition expectation operator satisfies

`E[V(X₁) | X₀=x] - V(x) ≤ -η`

outside a set `C`, with `η > 0` and `V ≥ 0`, then the truncated expected
hitting times of `C` are uniformly bounded:

`E_x[τ_C ∧ n] ≤ V(x) / η`.

This is the real algebraic content normally proved by stopped
supermartingales / optional stopping.  The interface below is intentionally
an expectation-operator interface: it can be instantiated from mathlib
`Kernel` integrals later, while the scheduler proof can already use the
recurrence theorem without asserting measure-theoretic facts that have not
been formalised.
-/

noncomputable section

set_option linter.unusedSectionVars false
set_option linter.unusedVariables false

namespace Scheduleurm

/-- A one-step Markov transition expectation operator.

For a genuine Markov chain this is `step x f = E[f(X₁) | X₀=x]`.  The fields
are exactly the finite algebraic properties used in the stopped
Foster-Lyapunov proof. -/
structure TransitionExpectation (X : Type*) where
  step : X → (X → ℝ) → ℝ
  monotone : ∀ x {f g : X → ℝ}, (∀ y, f y ≤ g y) → step x f ≤ step x g
  map_const : ∀ x c, step x (fun _ => c) = c
  map_add : ∀ x f g, step x (fun y => f y + g y) = step x f + step x g
  map_smul : ∀ x c f, step x (fun y => c * f y) = c * step x f

namespace TransitionExpectation

variable {X : Type*} (K : TransitionExpectation X)

/-- Positivity of one-step expectation follows from monotonicity and
constant preservation. -/
theorem step_nonneg {f : X → ℝ} (x : X)
    (hf : ∀ y, 0 ≤ f y) :
    0 ≤ K.step x f := by
  have hmono : K.step x (fun _ : X => 0) ≤ K.step x f :=
    K.monotone x (fun y => hf y)
  rw [K.map_const x 0] at hmono
  exact hmono

end TransitionExpectation

variable {X : Type*}

/-- One-step Lyapunov drift under a transition expectation operator. -/
def expectedLyapunovDrift
    (K : TransitionExpectation X) (V : X → ℝ) (x : X) : ℝ :=
  K.step x V - V x

/-- Expected number of pre-hit steps up to horizon `n`.

This is the dynamic-programming recursion for `E_x[τ_C ∧ n]`, where
`τ_C = inf {t ≥ 0 : X_t ∈ C}` and a chain starting inside `C` has hitting
time zero. -/
def expectedPreHitSteps
    (K : TransitionExpectation X) (C : X → Prop) [DecidablePred C] :
    ℕ → X → ℝ
  | 0, _ => 0
  | n + 1, x =>
      if C x then 0 else 1 + K.step x (expectedPreHitSteps K C n)

/-- Expected stopped Lyapunov value up to horizon `n`: the dynamic-programming
recursion for `E_x[V(X_{τ_C ∧ n})]`. -/
def expectedStoppedLyapunov
    (K : TransitionExpectation X) (C : X → Prop) [DecidablePred C]
    (V : X → ℝ) : ℕ → X → ℝ
  | 0, x => V x
  | n + 1, x =>
      if C x then V x else K.step x (expectedStoppedLyapunov K C V n)

/-- Finite expected hitting time to `C`, expressed through uniformly bounded
truncations.  In a measure-theoretic instantiation, this corresponds to
`sup_n E[τ_C ∧ n] < ∞`, hence `E[τ_C] < ∞` by monotone convergence. -/
def FiniteExpectedHittingTimeToSet
    (K : TransitionExpectation X) (C : X → Prop) [DecidablePred C]
    (x : X) : Prop :=
  ∃ M : ℝ, ∀ n : ℕ, expectedPreHitSteps K C n x ≤ M

/-- Expected return time to `C`, with return counted after at least one
transition.  This is the dynamic-programming recursion for
`E_x[τ_C⁺ ∧ n]`: at horizon `n + 1`, the chain pays one step and then pays
the pre-hit time to `C` from the next state. -/
def expectedReturnSteps
    (K : TransitionExpectation X) (C : X → Prop) [DecidablePred C] :
    ℕ → X → ℝ
  | 0, _ => 0
  | n + 1, x => 1 + K.step x (expectedPreHitSteps K C n)

/-- Finite expected return time to a set, again expressed through uniformly
bounded truncations. -/
def FiniteExpectedReturnTimeToSet
    (K : TransitionExpectation X) (C : X → Prop) [DecidablePred C]
    (x : X) : Prop :=
  ∃ M : ℝ, ∀ n : ℕ, expectedReturnSteps K C n x ≤ M

/-- A predicate denotes a finite set when its subtype is finite.  This is the
right boundary for countable-state Markov-chain positive recurrence: a
Foster drift theorem hits a finite/small set; local return assumptions on that
set then give the usual recurrent class conclusion. -/
def PredicateFinite (C : X → Prop) : Prop :=
  Finite {x : X // C x}

/-- Positive recurrence through a finite set.  This is a deliberately
precise, non-fake positive-recurrence notion for the expectation-operator
interface: every state has finite expected hitting time to a finite set `C`,
and every state inside `C` has finite expected return time to `C`.

For irreducible countable Markov chains this is the standard finite-small-set
form used on the way to state positive recurrence. -/
def PositiveRecurrentViaFiniteSet
    (K : TransitionExpectation X) (C : X → Prop) [DecidablePred C] : Prop :=
  PredicateFinite C ∧ (∃ x : X, C x) ∧
    (∀ x : X, FiniteExpectedHittingTimeToSet K C x) ∧
    (∀ x : X, C x → FiniteExpectedReturnTimeToSet K C x)

/-- A Foster drift certificate for hitting a set `C`. -/
def FosterHittingTimeCertificate
    (K : TransitionExpectation X) (C : X → Prop) [DecidablePred C]
    (V : X → ℝ) (η : ℝ) : Prop :=
  0 < η ∧ (∀ x, 0 ≤ V x) ∧
    ∀ x, ¬ C x → expectedLyapunovDrift K V x ≤ -η

/-- Stopped Foster-Lyapunov supermartingale inequality. -/
theorem stoppedLyapunov_add_eta_preHit_le
    (K : TransitionExpectation X) (C : X → Prop) [DecidablePred C]
    (V : X → ℝ) {η : ℝ}
    (hη : 0 < η)
    (hdrift : ∀ x, ¬ C x → expectedLyapunovDrift K V x ≤ -η) :
    ∀ n x,
      expectedStoppedLyapunov K C V n x
        + η * expectedPreHitSteps K C n x ≤ V x := by
  intro n
  induction n with
  | zero =>
      intro x
      simp [expectedStoppedLyapunov, expectedPreHitSteps]
  | succ n ih =>
      intro x
      by_cases hx : C x
      · simp [expectedStoppedLyapunov, expectedPreHitSteps, hx]
      · simp [expectedStoppedLyapunov, expectedPreHitSteps, hx]
        have hmono :
            K.step x
                (fun y =>
                  expectedStoppedLyapunov K C V n y
                    + η * expectedPreHitSteps K C n y)
              ≤ K.step x V :=
          K.monotone x (fun y => ih y)
        have hinside :
            K.step x (expectedStoppedLyapunov K C V n)
                + η * K.step x (expectedPreHitSteps K C n)
              ≤ K.step x V := by
          rw [K.map_add, K.map_smul] at hmono
          exact hmono
        have hdr := hdrift x hx
        unfold expectedLyapunovDrift at hdr
        nlinarith

/-- Stopped Lyapunov values remain nonnegative when `V ≥ 0`. -/
theorem expectedStoppedLyapunov_nonneg
    (K : TransitionExpectation X) (C : X → Prop) [DecidablePred C]
    (V : X → ℝ)
    (hV : ∀ x, 0 ≤ V x) :
    ∀ n x, 0 ≤ expectedStoppedLyapunov K C V n x := by
  intro n
  induction n with
  | zero =>
      intro x
      simpa [expectedStoppedLyapunov] using hV x
  | succ n ih =>
      intro x
      by_cases hx : C x
      · simp [expectedStoppedLyapunov, hx, hV x]
      · simp [expectedStoppedLyapunov, hx]
        exact K.step_nonneg x (fun y => ih y)

/-- Foster negative drift gives a uniform bound on truncated expected hitting
times: `E_x[τ_C ∧ n] ≤ V(x)/η`. -/
theorem expectedPreHitSteps_le_div
    (K : TransitionExpectation X) (C : X → Prop) [DecidablePred C]
    (V : X → ℝ) {η : ℝ}
    (hη : 0 < η)
    (hV : ∀ x, 0 ≤ V x)
    (hdrift : ∀ x, ¬ C x → expectedLyapunovDrift K V x ≤ -η) :
    ∀ n x, expectedPreHitSteps K C n x ≤ V x / η := by
  intro n x
  have hmain := stoppedLyapunov_add_eta_preHit_le K C V hη hdrift n x
  have hstop := expectedStoppedLyapunov_nonneg K C V hV n x
  rw [le_div_iff₀ hη]
  nlinarith

/-- Foster negative drift implies finite expected hitting time to `C`, in the
standard uniformly-bounded-truncations sense. -/
theorem foster_negative_drift_implies_finite_expected_hitting_time
    (K : TransitionExpectation X) (C : X → Prop) [DecidablePred C]
    (V : X → ℝ) {η : ℝ}
    (hη : 0 < η)
    (hV : ∀ x, 0 ≤ V x)
    (hdrift : ∀ x, ¬ C x → expectedLyapunovDrift K V x ≤ -η) :
    ∀ x, FiniteExpectedHittingTimeToSet K C x := by
  intro x
  refine ⟨V x / η, ?_⟩
  intro n
  exact expectedPreHitSteps_le_div K C V hη hV hdrift n x

/-- The same Foster drift certificate also gives finite expected return time
to `C` after one transition.  This discharges the local-return obligation
whenever the model has already supplied a finite one-step Lyapunov
expectation through the `TransitionExpectation` operator. -/
theorem foster_negative_drift_implies_finite_expected_return_time_to_set
    (K : TransitionExpectation X) (C : X → Prop) [DecidablePred C]
    (V : X → ℝ) {η : ℝ}
    (hη : 0 < η)
    (hV : ∀ x, 0 ≤ V x)
    (hdrift : ∀ x, ¬ C x → expectedLyapunovDrift K V x ≤ -η) :
    ∀ x, FiniteExpectedReturnTimeToSet K C x := by
  intro x
  refine ⟨max 0 (1 + (1 / η) * K.step x V), ?_⟩
  intro n
  cases n with
  | zero =>
      change (0 : ℝ) ≤ max 0 (1 + (1 / η) * K.step x V)
      exact le_max_left 0 (1 + (1 / η) * K.step x V)
  | succ n =>
      change 1 + K.step x (expectedPreHitSteps K C n)
        ≤ max 0 (1 + (1 / η) * K.step x V)
      have hpre : K.step x (expectedPreHitSteps K C n)
          ≤ K.step x (fun y => V y / η) :=
        K.monotone x
          (fun y => expectedPreHitSteps_le_div K C V hη hV hdrift n y)
      have hstep_div : K.step x (fun y => V y / η)
          = (1 / η) * K.step x V := by
        have hfun : (fun y => V y / η) = (fun y => (1 / η) * V y) := by
          funext y
          ring
        rw [hfun, K.map_smul]
      have hb : 1 + K.step x (expectedPreHitSteps K C n)
          ≤ 1 + (1 / η) * K.step x V := by
        linarith
      exact le_trans hb (le_max_right 0 (1 + (1 / η) * K.step x V))

/-- Certificate form of the Foster hitting-time theorem. -/
theorem FosterHittingTimeCertificate.finiteExpectedHittingTime
    (K : TransitionExpectation X) (C : X → Prop) [DecidablePred C]
    (V : X → ℝ) {η : ℝ}
    (hcert : FosterHittingTimeCertificate K C V η) :
    ∀ x, FiniteExpectedHittingTimeToSet K C x := by
  rcases hcert with ⟨hη, hV, hdrift⟩
  exact foster_negative_drift_implies_finite_expected_hitting_time
    K C V hη hV hdrift

/-- Foster hitting-time control plus finite-set and local-return hypotheses
gives the countable-chain positive-recurrence conclusion through `C`.

The local return assumption is intentionally explicit: it is the exact extra
irreducibility/small-set part that a real Markov-chain model must supply
after the Lyapunov drift calculation. -/
theorem foster_certificate_positive_recurrent_via_finite_set
    (K : TransitionExpectation X) (C : X → Prop) [DecidablePred C]
    (V : X → ℝ) {η : ℝ}
    (hfinite : PredicateFinite C)
    (hnonempty : ∃ x : X, C x)
    (hreturn : ∀ x : X, C x → FiniteExpectedReturnTimeToSet K C x)
    (hcert : FosterHittingTimeCertificate K C V η) :
    PositiveRecurrentViaFiniteSet K C := by
  refine ⟨hfinite, hnonempty, ?_, hreturn⟩
  exact FosterHittingTimeCertificate.finiteExpectedHittingTime K C V hcert

/-- Direct theorem form of finite-set positive recurrence from Foster
negative drift and local return. -/
theorem foster_negative_drift_positive_recurrent_via_finite_set
    (K : TransitionExpectation X) (C : X → Prop) [DecidablePred C]
    (V : X → ℝ) {η : ℝ}
    (hfinite : PredicateFinite C)
    (hnonempty : ∃ x : X, C x)
    (hreturn : ∀ x : X, C x → FiniteExpectedReturnTimeToSet K C x)
    (hη : 0 < η)
    (hV : ∀ x, 0 ≤ V x)
    (hdrift : ∀ x, ¬ C x → expectedLyapunovDrift K V x ≤ -η) :
    PositiveRecurrentViaFiniteSet K C := by
  refine ⟨hfinite, hnonempty, ?_, hreturn⟩
  exact foster_negative_drift_implies_finite_expected_hitting_time
    K C V hη hV hdrift

/-- Foster negative drift outside a finite nonempty set is enough to prove
the finite-set positive-recurrence certificate in this expectation-operator
interface.  The local-return part follows from the same stopped Lyapunov
bound after one transition. -/
theorem foster_negative_drift_positive_recurrent_via_finite_set'
    (K : TransitionExpectation X) (C : X → Prop) [DecidablePred C]
    (V : X → ℝ) {η : ℝ}
    (hfinite : PredicateFinite C)
    (hnonempty : ∃ x : X, C x)
    (hη : 0 < η)
    (hV : ∀ x, 0 ≤ V x)
    (hdrift : ∀ x, ¬ C x → expectedLyapunovDrift K V x ≤ -η) :
    PositiveRecurrentViaFiniteSet K C := by
  refine ⟨hfinite, hnonempty, ?_, ?_⟩
  · exact foster_negative_drift_implies_finite_expected_hitting_time
      K C V hη hV hdrift
  · intro x _hx
    exact foster_negative_drift_implies_finite_expected_return_time_to_set
      K C V hη hV hdrift x

end Scheduleurm


/-! ## Source: Scheduleurm/QueueRecurrence.lean -/


/-!
# Scheduleurm: queue-state recurrence bridge

This file instantiates the abstract Foster hitting-time theorem for queue
states.  The Markov state space is the nonnegative queue cone:

`QueueState I = {Q : ServiceVec I // Nonnegative Q}`.

The result is the rigorous recurrence bridge used by the scheduler theorem:
if the one-step expected Lyapunov drift is bounded by

`B - η ||Q||₁`

with `η > 0`, then the chain has finite expected hitting time to every
large enough `||Q||₁` sublevel set.  The robust candidate MaxWeight theorem
supplies that linear drift bound once the actual Markov transition drift is
dominated by the deterministic queue-drift inequality.
-/

noncomputable section

set_option linter.unusedSectionVars false
set_option linter.unusedVariables false

namespace Scheduleurm

variable {I A : Type*} [Fintype I] [DecidableEq A]

/-- The queue Markov-chain state space: queue vectors are always
coordinatewise nonnegative. -/
abbrev QueueState (I : Type*) [Fintype I] :=
  {Q : ServiceVec I // Nonnegative Q}

/-- Queue-state Lyapunov function. -/
def queueLyapunov (Q : QueueState I) : ℝ :=
  Lyapunov Q.val

/-- Queue-state total backlog. -/
def queueL1 (Q : QueueState I) : ℝ :=
  l1 Q.val

/-- A finite/backlog-small set used in Foster recurrence arguments. -/
def queueSmallSet (R : ℝ) (Q : QueueState I) : Prop :=
  queueL1 Q ≤ R

instance queueSmallSetDecidablePred (R : ℝ) :
    DecidablePred (queueSmallSet (I := I) R) :=
  Classical.decPred _

/-- Quadratic Lyapunov is nonnegative. -/
lemma lyapunov_nonneg (Q : ServiceVec I) :
    0 ≤ Lyapunov Q := by
  unfold Lyapunov
  exact Finset.sum_nonneg (fun i _ => by positivity)

/-- Queue-state Lyapunov is nonnegative. -/
lemma queueLyapunov_nonneg (Q : QueueState I) :
    0 ≤ queueLyapunov Q := by
  exact lyapunov_nonneg Q.val

/-- A linear `B - η||Q||₁` expected drift bound gives strict negative drift
outside any sublevel set with radius large enough to absorb `B` plus the
desired margin `α`. -/
theorem linear_l1_drift_outside_small_set
    (K : TransitionExpectation (QueueState I))
    {B η α R : ℝ}
    (hη : 0 < η)
    (hR : B + α ≤ η * R)
    (hdrift : ∀ Q : QueueState I,
      expectedLyapunovDrift K (queueLyapunov (I := I)) Q
        ≤ B - η * queueL1 Q) :
    ∀ Q : QueueState I, ¬ queueSmallSet R Q →
      expectedLyapunovDrift K (queueLyapunov (I := I)) Q ≤ -α := by
  intro Q hsmall
  have houtside : R < queueL1 Q := lt_of_not_ge hsmall
  have hlinear := hdrift Q
  have hmul : η * R < η * queueL1 Q :=
    mul_lt_mul_of_pos_left houtside hη
  nlinarith

/-- Linear negative backlog drift implies finite expected hitting time to a
large enough backlog sublevel set. -/
theorem linear_l1_drift_finite_expected_hitting_time
    (K : TransitionExpectation (QueueState I))
    {B η α R : ℝ}
    (hη : 0 < η) (hα : 0 < α)
    (hR : B + α ≤ η * R)
    (hdrift : ∀ Q : QueueState I,
      expectedLyapunovDrift K (queueLyapunov (I := I)) Q
        ≤ B - η * queueL1 Q) :
    ∀ Q0 : QueueState I,
      FiniteExpectedHittingTimeToSet K (queueSmallSet (I := I) R) Q0 := by
  have houtside :
      ∀ Q : QueueState I, ¬ queueSmallSet (I := I) R Q →
        expectedLyapunovDrift K (queueLyapunov (I := I)) Q ≤ -α :=
    linear_l1_drift_outside_small_set K hη hR hdrift
  exact foster_negative_drift_implies_finite_expected_hitting_time
    K (queueSmallSet (I := I) R) (queueLyapunov (I := I))
    hα (fun Q => queueLyapunov_nonneg Q) houtside

/-- Robust candidate MaxWeight, interpreted as a Markov-chain expected drift
bound, implies finite expected hitting time to a sufficiently large backlog
sublevel set.

The hypothesis `hdominated` is the precise probabilistic boundary: it is where
a concrete stochastic service/arrival model must prove that the conditional
expected Lyapunov drift is no larger than the deterministic queue-drift
expression already verified in `RobustPolicy`. -/
theorem robust_candidate_markov_finite_expected_hitting_time
    (K : TransitionExpectation (QueueState I))
    (full cand : ActionFamily A)
    (μtrue lower : A → ServiceVec I)
    (lam : ServiceVec I) {δ εcand εest B Pmax α R : ℝ}
    (hlam : Nonnegative lam)
    (hcap : InCapacityWithSlack full μtrue lam δ)
    (hgap : SupportGapAtMost full cand μtrue εcand)
    (penalty : ServiceVec I → A → ℝ)
    (astar : ServiceVec I → A)
    (hlower_gap : ∀ Q : ServiceVec I, Nonnegative Q →
      support cand μtrue Q ≤ support cand lower Q + εest * l1 Q)
    (hmax : ∀ Q : ServiceVec I, Nonnegative Q →
      RobustScoreMaximizer cand lower Q (penalty Q) (astar Q))
    (hpen : ∀ Q : ServiceVec I, Nonnegative Q →
      PenaltyBounded cand (penalty Q) Pmax)
    (hSecond : ∀ Q : ServiceVec I, Nonnegative Q →
      secondOrderTerm lam (lower (astar Q)) ≤ B)
    (hlower_nonneg : ∀ Q : ServiceVec I, Nonnegative Q →
      Nonnegative (lower (astar Q)))
    (hdominated : ∀ Q : QueueState I,
      expectedLyapunovDrift K (queueLyapunov (I := I)) Q
        ≤ Lyapunov (queueStep Q.val lam (lower (astar Q.val)))
            - Lyapunov Q.val)
    (hmargin : 0 < δ - (εcand + εest))
    (hα : 0 < α)
    (hR : B + Pmax + α ≤ (δ - (εcand + εest)) * R) :
    ∀ Q0 : QueueState I,
      FiniteExpectedHittingTimeToSet K (queueSmallSet (I := I) R) Q0 := by
  have hfoster : FosterNegativeDrift
      (fun Q =>
        Lyapunov (queueStep Q lam (lower (astar Q))) - Lyapunov Q)
      (B + Pmax) (δ - (εcand + εest)) :=
    robust_candidate_policy_foster_negative_drift
      full cand μtrue lower lam hlam hcap hgap penalty astar
      hlower_gap hmax hpen hSecond hlower_nonneg rfl
  have hlinear : ∀ Q : QueueState I,
      expectedLyapunovDrift K (queueLyapunov (I := I)) Q
        ≤ (B + Pmax) - (δ - (εcand + εest)) * queueL1 Q := by
    intro Q
    have hdet := hfoster Q.val Q.property
    have hdom := hdominated Q
    exact le_trans hdom hdet
  exact linear_l1_drift_finite_expected_hitting_time
    K hmargin hα hR hlinear

end Scheduleurm


/-! ## Source: Scheduleurm/IntegerQueue.lean -/


/-!
# Scheduleurm: countable/integer queue state space

This file addresses the countable-state boundary in the recurrence theorem.
The real-valued queue cone is useful for algebraic drift inequalities, but a
countable-state Markov-chain positive-recurrence statement should use integer
queues.  We therefore provide:

* integer queue states `I → ℕ`;
* finite backlog sublevel sets;
* Foster hitting and positive-recurrence-through-a-finite-set theorems for
  integer queues.
-/

noncomputable section

set_option linter.unusedSectionVars false
set_option linter.unusedVariables false

namespace Scheduleurm

open BigOperators

variable {I : Type*} [Fintype I] [DecidableEq I]

/-- Countable queue state space for integer-valued job queues. -/
abbrev NatQueueState (I : Type*) :=
  I → ℕ

/-- Integer queue embedded as a real service vector. -/
def natQueueToServiceVec (Q : NatQueueState I) : ServiceVec I :=
  fun i => (Q i : ℝ)

/-- Integer total backlog. -/
def natQueueL1Nat (Q : NatQueueState I) : ℕ :=
  ∑ i : I, Q i

/-- Real-valued total backlog of an integer queue. -/
def natQueueL1 (Q : NatQueueState I) : ℝ :=
  (natQueueL1Nat Q : ℝ)

/-- Integer-queue Lyapunov function, reusing the real quadratic Lyapunov. -/
def natQueueLyapunov (Q : NatQueueState I) : ℝ :=
  Lyapunov (natQueueToServiceVec Q)

/-- Backlog sublevel set for integer queues. -/
def natQueueSmallSet (N : ℕ) (Q : NatQueueState I) : Prop :=
  natQueueL1Nat Q ≤ N

instance natQueueSmallSetDecidablePred (N : ℕ) :
    DecidablePred (natQueueSmallSet (I := I) N) :=
  Classical.decPred _

/-- Each queue coordinate is bounded by the total integer backlog. -/
lemma natQueue_coord_le_l1 (Q : NatQueueState I) (i : I) :
    Q i ≤ natQueueL1Nat Q := by
  unfold natQueueL1Nat
  exact Finset.single_le_sum (fun _ _ => Nat.zero_le _) (Finset.mem_univ i)

/-- Embeds the integer sublevel set into a finite product of bounded
coordinates. -/
def natQueueSublevelEmbedding (N : ℕ) :
    {Q : NatQueueState I // natQueueSmallSet (I := I) N Q}
      ↪ (I → Fin (N + 1)) where
  toFun Q := fun i =>
    ⟨Q.val i,
      Nat.lt_succ_of_le
        (le_trans (natQueue_coord_le_l1 Q.val i) Q.property)⟩
  inj' := by
    intro Q R h
    apply Subtype.ext
    funext i
    exact Fin.ext_iff.mp (congr_fun h i)

/-- Integer backlog sublevel sets are finite. -/
instance natQueueSublevelFintype (N : ℕ) :
    Fintype {Q : NatQueueState I // natQueueSmallSet (I := I) N Q} :=
  Fintype.ofInjective (natQueueSublevelEmbedding (I := I) N)
    (natQueueSublevelEmbedding (I := I) N).injective

/-- The finite-set predicate for integer queue sublevels. -/
theorem natQueueSmallSet_finite (N : ℕ) :
    PredicateFinite (natQueueSmallSet (I := I) N) := by
  unfold PredicateFinite
  infer_instance

/-- The zero queue lies in every integer sublevel set. -/
theorem natQueueSmallSet_nonempty (N : ℕ) :
    ∃ Q : NatQueueState I, natQueueSmallSet (I := I) N Q := by
  refine ⟨fun _ => 0, ?_⟩
  simp [natQueueSmallSet, natQueueL1Nat]

/-- Integer-queue Lyapunov is nonnegative. -/
lemma natQueueLyapunov_nonneg (Q : NatQueueState I) :
    0 ≤ natQueueLyapunov Q := by
  unfold natQueueLyapunov
  exact lyapunov_nonneg (natQueueToServiceVec Q)

/-- Linear `B - η||Q||₁` drift is strictly negative outside a sufficiently
large integer backlog sublevel set. -/
theorem natQueue_linear_drift_outside_small_set
    (K : TransitionExpectation (NatQueueState I))
    {B η α : ℝ} {N : ℕ}
    (hη : 0 < η)
    (hN : B + α ≤ η * (N : ℝ))
    (hdrift : ∀ Q : NatQueueState I,
      expectedLyapunovDrift K (natQueueLyapunov (I := I)) Q
        ≤ B - η * natQueueL1 Q) :
    ∀ Q : NatQueueState I, ¬ natQueueSmallSet (I := I) N Q →
      expectedLyapunovDrift K (natQueueLyapunov (I := I)) Q ≤ -α := by
  intro Q hsmall
  have houtNat : N < natQueueL1Nat Q := Nat.lt_of_not_ge hsmall
  have hout : (N : ℝ) < natQueueL1 Q := by
    unfold natQueueL1
    exact_mod_cast houtNat
  have hlinear := hdrift Q
  have hmul : η * (N : ℝ) < η * natQueueL1 Q :=
    mul_lt_mul_of_pos_left hout hη
  nlinarith

/-- Integer-queue Foster drift gives finite expected hitting time to a finite
backlog sublevel set. -/
theorem natQueue_linear_drift_finite_expected_hitting_time
    (K : TransitionExpectation (NatQueueState I))
    {B η α : ℝ} {N : ℕ}
    (hη : 0 < η) (hα : 0 < α)
    (hN : B + α ≤ η * (N : ℝ))
    (hdrift : ∀ Q : NatQueueState I,
      expectedLyapunovDrift K (natQueueLyapunov (I := I)) Q
        ≤ B - η * natQueueL1 Q) :
    ∀ Q0 : NatQueueState I,
      FiniteExpectedHittingTimeToSet K (natQueueSmallSet (I := I) N) Q0 := by
  have houtside :
      ∀ Q : NatQueueState I, ¬ natQueueSmallSet (I := I) N Q →
        expectedLyapunovDrift K (natQueueLyapunov (I := I)) Q ≤ -α :=
    natQueue_linear_drift_outside_small_set K hη hN hdrift
  exact foster_negative_drift_implies_finite_expected_hitting_time
    K (natQueueSmallSet (I := I) N) (natQueueLyapunov (I := I))
    hα (fun Q => natQueueLyapunov_nonneg Q) houtside

/-- Integer-queue Foster drift plus finite local return on the small set gives
the standard countable-state positive-recurrence-through-a-finite-set
conclusion. -/
theorem natQueue_linear_drift_positive_recurrent_via_finite_set
    (K : TransitionExpectation (NatQueueState I))
    {B η α : ℝ} {N : ℕ}
    (hη : 0 < η) (hα : 0 < α)
    (hN : B + α ≤ η * (N : ℝ))
    (hdrift : ∀ Q : NatQueueState I,
      expectedLyapunovDrift K (natQueueLyapunov (I := I)) Q
        ≤ B - η * natQueueL1 Q)
    (hreturn : ∀ Q : NatQueueState I,
      natQueueSmallSet (I := I) N Q →
        FiniteExpectedReturnTimeToSet K (natQueueSmallSet (I := I) N) Q) :
    PositiveRecurrentViaFiniteSet K (natQueueSmallSet (I := I) N) := by
  have houtside :
      ∀ Q : NatQueueState I, ¬ natQueueSmallSet (I := I) N Q →
        expectedLyapunovDrift K (natQueueLyapunov (I := I)) Q ≤ -α :=
    natQueue_linear_drift_outside_small_set K hη hN hdrift
  exact foster_negative_drift_positive_recurrent_via_finite_set
    K (natQueueSmallSet (I := I) N) (natQueueLyapunov (I := I))
    (natQueueSmallSet_finite N) (natQueueSmallSet_nonempty N) hreturn
    hα (fun Q => natQueueLyapunov_nonneg Q) houtside

/-- Integer-queue Foster drift gives positive recurrence through a finite
backlog sublevel set without a separate local-return hypothesis.  The return
bound is derived from the same one-step Lyapunov drift certificate. -/
theorem natQueue_linear_drift_positive_recurrent_via_finite_set'
    (K : TransitionExpectation (NatQueueState I))
    {B η α : ℝ} {N : ℕ}
    (hη : 0 < η) (hα : 0 < α)
    (hN : B + α ≤ η * (N : ℝ))
    (hdrift : ∀ Q : NatQueueState I,
      expectedLyapunovDrift K (natQueueLyapunov (I := I)) Q
        ≤ B - η * natQueueL1 Q) :
    PositiveRecurrentViaFiniteSet K (natQueueSmallSet (I := I) N) := by
  have houtside :
      ∀ Q : NatQueueState I, ¬ natQueueSmallSet (I := I) N Q →
        expectedLyapunovDrift K (natQueueLyapunov (I := I)) Q ≤ -α :=
    natQueue_linear_drift_outside_small_set K hη hN hdrift
  exact foster_negative_drift_positive_recurrent_via_finite_set'
    K (natQueueSmallSet (I := I) N) (natQueueLyapunov (I := I))
    (natQueueSmallSet_finite N) (natQueueSmallSet_nonempty N)
    hα (fun Q => natQueueLyapunov_nonneg Q) houtside

end Scheduleurm


/-! ## Source: Scheduleurm/StochasticQueueModel.lean -/


/-!
# Scheduleurm: stochastic queue-model drift boundary

The deterministic MaxWeight drift proof bounds

`V([Q-S]^+ + A) - V(Q)`.

A real stochastic queueing model must additionally prove that its conditional
one-step expected Lyapunov drift is dominated by that deterministic bound.
This file names that boundary and provides the exact bridge theorem.  The
probabilistic model-specific work is intentionally isolated in the
`drift_dominated` field; no positive-recurrence conclusion is claimed without
it.
-/

noncomputable section

set_option linter.unusedSectionVars false
set_option linter.unusedVariables false

namespace Scheduleurm

variable {I A : Type*} [Fintype I] [DecidableEq I] [DecidableEq A]

/-- A real-valued queue transition model together with the exact conditional
drift-domination obligation needed to use the deterministic scheduler proof. -/
structure RealQueueTransitionModel (I : Type*) [Fintype I] where
  K : TransitionExpectation (QueueState I)
  deterministicDrift : QueueState I → ℝ
  drift_dominated :
    ∀ Q : QueueState I,
      expectedLyapunovDrift K (queueLyapunov (I := I)) Q
        ≤ deterministicDrift Q

/-- Integer-valued countable-state queue transition model. -/
structure NatQueueTransitionModel (I : Type*) [Fintype I] where
  K : TransitionExpectation (NatQueueState I)
  deterministicDrift : NatQueueState I → ℝ
  drift_dominated :
    ∀ Q : NatQueueState I,
      expectedLyapunovDrift K (natQueueLyapunov (I := I)) Q
        ≤ deterministicDrift Q

/-- If the stochastic model's conditional drift is dominated by a linear
negative backlog bound, then it has finite expected hitting time to a large
sublevel set. -/
theorem real_model_linear_drift_finite_expected_hitting_time
    (M : RealQueueTransitionModel I)
    {B η α R : ℝ}
    (hη : 0 < η) (hα : 0 < α)
    (hR : B + α ≤ η * R)
    (hdet : ∀ Q : QueueState I,
      M.deterministicDrift Q ≤ B - η * queueL1 Q) :
    ∀ Q0 : QueueState I,
      FiniteExpectedHittingTimeToSet M.K (queueSmallSet (I := I) R) Q0 := by
  have hdrift : ∀ Q : QueueState I,
      expectedLyapunovDrift M.K (queueLyapunov (I := I)) Q
        ≤ B - η * queueL1 Q := by
    intro Q
    exact le_trans (M.drift_dominated Q) (hdet Q)
  exact linear_l1_drift_finite_expected_hitting_time
    M.K hη hα hR hdrift

/-- Integer-valued model version: conditional drift domination plus local
return on the finite sublevel set gives the countable-state positive
recurrence conclusion through that finite set. -/
theorem nat_model_linear_drift_positive_recurrent_via_finite_set
    (M : NatQueueTransitionModel I)
    {B η α : ℝ} {N : ℕ}
    (hη : 0 < η) (hα : 0 < α)
    (hN : B + α ≤ η * (N : ℝ))
    (hdet : ∀ Q : NatQueueState I,
      M.deterministicDrift Q ≤ B - η * natQueueL1 Q)
    (hreturn : ∀ Q : NatQueueState I,
      natQueueSmallSet (I := I) N Q →
        FiniteExpectedReturnTimeToSet M.K (natQueueSmallSet (I := I) N) Q) :
    PositiveRecurrentViaFiniteSet M.K (natQueueSmallSet (I := I) N) := by
  have hdrift : ∀ Q : NatQueueState I,
      expectedLyapunovDrift M.K (natQueueLyapunov (I := I)) Q
        ≤ B - η * natQueueL1 Q := by
    intro Q
    exact le_trans (M.drift_dominated Q) (hdet Q)
  exact natQueue_linear_drift_positive_recurrent_via_finite_set
    M.K hη hα hN hdrift hreturn

/-- Integer-valued model version with local return discharged by the same
Foster-Lyapunov drift bound. -/
theorem nat_model_linear_drift_positive_recurrent_via_finite_set'
    (M : NatQueueTransitionModel I)
    {B η α : ℝ} {N : ℕ}
    (hη : 0 < η) (hα : 0 < α)
    (hN : B + α ≤ η * (N : ℝ))
    (hdet : ∀ Q : NatQueueState I,
      M.deterministicDrift Q ≤ B - η * natQueueL1 Q) :
    PositiveRecurrentViaFiniteSet M.K (natQueueSmallSet (I := I) N) := by
  have hdrift : ∀ Q : NatQueueState I,
      expectedLyapunovDrift M.K (natQueueLyapunov (I := I)) Q
        ≤ B - η * natQueueL1 Q := by
    intro Q
    exact le_trans (M.drift_dominated Q) (hdet Q)
  exact natQueue_linear_drift_positive_recurrent_via_finite_set'
    M.K hη hα hN hdrift

end Scheduleurm


/-! ## Source: Scheduleurm/ConcreteStochasticModel.lean -/


/-!
# Scheduleurm: a concrete finite-support stochastic queue model

This file discharges the `drift_dominated` obligation for an explicit
arrival/service model.

At each queue state `Q`, a finite random outcome `ω : Ω` is drawn with
probability `prob Q ω`.  The outcome supplies an integer arrival batch
`arrivals Q ω` and an integer service realization `service Q ω`, and the
queue evolves by

`Q⁺ᵢ = Qᵢ - serviceᵢ + arrivalsᵢ`,

where natural-number subtraction implements `[Qᵢ - serviceᵢ]^+`.

The model is intentionally finite-support rather than measure-theoretic:
it is concrete enough to verify the queueing drift calculation used by the
paper, while matching the expectation-operator recurrence interface already
used in the Foster-Lyapunov layer.
-/

noncomputable section

set_option linter.unusedSectionVars false
set_option linter.unusedVariables false

namespace Scheduleurm

open BigOperators

variable {X I Ω : Type*} [Fintype I] [DecidableEq I] [Fintype Ω]

/-! ## Finite-support transition expectations -/

/-- Finite-support one-step expectation operator. -/
def finiteSupportTransitionExpectation
    (prob : X → Ω → ℝ) (next : X → Ω → X)
    (hprob_nonneg : ∀ x ω, 0 ≤ prob x ω)
    (hprob_sum : ∀ x, ∑ ω : Ω, prob x ω = 1) :
    TransitionExpectation X where
  step x f := ∑ ω : Ω, prob x ω * f (next x ω)
  monotone := by
    intro x f g hfg
    apply Finset.sum_le_sum
    intro ω _
    exact mul_le_mul_of_nonneg_left (hfg (next x ω)) (hprob_nonneg x ω)
  map_const := by
    intro x c
    calc
      (∑ ω : Ω, prob x ω * c) = (∑ ω : Ω, prob x ω) * c := by
        rw [Finset.sum_mul]
      _ = c := by rw [hprob_sum x, one_mul]
  map_add := by
    intro x f g
    rw [← Finset.sum_add_distrib]
    apply Finset.sum_congr rfl
    intro ω _
    ring
  map_smul := by
    intro x c f
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro ω _
    ring

/-! ## Integer queue dynamics -/

/-- Integer queue update.  Natural subtraction is truncated subtraction, so
this is the integer version of `[Q-S]^+ + A`. -/
def natQueueStep (Q Arr Serv : NatQueueState I) : NatQueueState I :=
  fun i => Q i - Serv i + Arr i

/-- Natural truncated subtraction, embedded in the reals, is `max(q-s,0)`. -/
lemma natCast_tsub_eq_max_sub (q s : ℕ) :
    ((q - s : ℕ) : ℝ) = max ((q : ℝ) - (s : ℝ)) 0 := by
  by_cases h : s ≤ q
  · have hsub : ((q - s : ℕ) : ℝ) = (q : ℝ) - (s : ℝ) := by
      exact Nat.cast_sub (R := ℝ) h
    have hsle : (s : ℝ) ≤ q := by exact_mod_cast h
    have hnonneg : 0 ≤ (q : ℝ) - (s : ℝ) := by linarith
    rw [hsub, max_eq_left hnonneg]
  · have hlt : q < s := Nat.lt_of_not_ge h
    have hsq : q ≤ s := Nat.le_of_lt hlt
    have hsub0 : q - s = 0 := Nat.sub_eq_zero_of_le hsq
    have hqs : (q : ℝ) ≤ s := by exact_mod_cast hsq
    have hnonpos : (q : ℝ) - (s : ℝ) ≤ 0 := by linarith
    rw [hsub0, max_eq_right hnonpos]
    norm_num

/-- Integer queue update agrees with the real queue update after embedding. -/
lemma natQueueStep_toServiceVec (Q Arr Serv : NatQueueState I) :
    natQueueToServiceVec (natQueueStep Q Arr Serv)
      = queueStep (natQueueToServiceVec Q) (natQueueToServiceVec Arr)
          (natQueueToServiceVec Serv) := by
  funext i
  unfold natQueueToServiceVec natQueueStep queueStep queueStepScalar
  rw [Nat.cast_add, natCast_tsub_eq_max_sub]

/-- Embedded integer vectors are nonnegative. -/
lemma natQueueToServiceVec_nonnegative (Q : NatQueueState I) :
    Nonnegative (natQueueToServiceVec Q) := by
  intro i
  exact Nat.cast_nonneg (Q i)

/-- The real `ℓ₁` norm of an embedded integer queue is its integer total
backlog. -/
lemma l1_natQueueToServiceVec (Q : NatQueueState I) :
    l1 (natQueueToServiceVec Q) = natQueueL1 Q := by
  unfold natQueueL1 natQueueL1Nat
  rw [l1_eq_sum_of_nonnegative (natQueueToServiceVec_nonnegative Q)]
  unfold natQueueToServiceVec
  simp

/-- One-sample integer queue drift satisfies the standard quadratic drift
inequality. -/
theorem natQueueStep_lyapunov_bound
    (Q Arr Serv : NatQueueState I) :
    natQueueLyapunov (natQueueStep Q Arr Serv) - natQueueLyapunov Q
      ≤ secondOrderTerm (natQueueToServiceVec Arr)
            (natQueueToServiceVec Serv)
          + dot (natQueueToServiceVec Q) (natQueueToServiceVec Arr)
          - dot (natQueueToServiceVec Q) (natQueueToServiceVec Serv) := by
  unfold natQueueLyapunov
  rw [natQueueStep_toServiceVec]
  exact lyapunov_queue_step_bound
    (Q := natQueueToServiceVec Q)
    (Arr := natQueueToServiceVec Arr)
    (Serv := natQueueToServiceVec Serv)
    (natQueueToServiceVec_nonnegative Q)
    (natQueueToServiceVec_nonnegative Arr)
    (natQueueToServiceVec_nonnegative Serv)

/-! ## Concrete finite-support stochastic queue model -/

/-- A concrete finite-support stochastic arrival/service queue model. -/
structure FiniteSupportQueueModel (I Ω : Type*) [Fintype I] [Fintype Ω] where
  prob : NatQueueState I → Ω → ℝ
  prob_nonneg : ∀ Q ω, 0 ≤ prob Q ω
  prob_sum : ∀ Q, ∑ ω : Ω, prob Q ω = 1
  arrivals : NatQueueState I → Ω → NatQueueState I
  service : NatQueueState I → Ω → NatQueueState I

namespace FiniteSupportQueueModel

variable (M : FiniteSupportQueueModel I Ω)

/-- The next integer queue state under outcome `ω`. -/
def next (Q : NatQueueState I) (ω : Ω) : NatQueueState I :=
  natQueueStep Q (M.arrivals Q ω) (M.service Q ω)

/-- One-step transition expectation induced by the finite arrival/service
model. -/
def K : TransitionExpectation (NatQueueState I) :=
  finiteSupportTransitionExpectation M.prob M.next M.prob_nonneg M.prob_sum

/-- One-sample Lyapunov drift. -/
def sampleLyapunovDrift (Q : NatQueueState I) (ω : Ω) : ℝ :=
  natQueueLyapunov (M.next Q ω) - natQueueLyapunov Q

/-- Exact finite-support conditional expected Lyapunov drift. -/
def expectedOneStepLyapunovDrift (Q : NatQueueState I) : ℝ :=
  ∑ ω : Ω, M.prob Q ω * M.sampleLyapunovDrift Q ω

/-- Conditional mean arrival vector. -/
def expectedArrival (Q : NatQueueState I) : ServiceVec I :=
  fun i => ∑ ω : Ω, M.prob Q ω * (M.arrivals Q ω i : ℝ)

/-- Conditional mean service vector. -/
def expectedService (Q : NatQueueState I) : ServiceVec I :=
  fun i => ∑ ω : Ω, M.prob Q ω * (M.service Q ω i : ℝ)

/-- Conditional expected quadratic second-order term in the queue drift. -/
def expectedSecondOrder (Q : NatQueueState I) : ℝ :=
  ∑ ω : Ω, M.prob Q ω *
    secondOrderTerm (natQueueToServiceVec (M.arrivals Q ω))
      (natQueueToServiceVec (M.service Q ω))

/-- The expected Lyapunov drift of the transition operator is exactly the
finite weighted drift over arrival/service outcomes. -/
theorem expectedLyapunovDrift_eq (Q : NatQueueState I) :
    expectedLyapunovDrift M.K (natQueueLyapunov (I := I)) Q
      = M.expectedOneStepLyapunovDrift Q := by
  unfold expectedLyapunovDrift K finiteSupportTransitionExpectation
    expectedOneStepLyapunovDrift sampleLyapunovDrift
  calc
    (∑ ω : Ω, M.prob Q ω * natQueueLyapunov (M.next Q ω))
        - natQueueLyapunov Q
        = (∑ ω : Ω, M.prob Q ω * natQueueLyapunov (M.next Q ω))
            - (∑ ω : Ω, M.prob Q ω) * natQueueLyapunov Q := by
          rw [M.prob_sum Q, one_mul]
    _ = (∑ ω : Ω, M.prob Q ω * natQueueLyapunov (M.next Q ω))
            - ∑ ω : Ω, M.prob Q ω * natQueueLyapunov Q := by
          rw [Finset.sum_mul]
    _ = ∑ ω : Ω,
            (M.prob Q ω * natQueueLyapunov (M.next Q ω)
              - M.prob Q ω * natQueueLyapunov Q) := by
          rw [Finset.sum_sub_distrib]
    _ = ∑ ω : Ω,
            M.prob Q ω
              * (natQueueLyapunov (M.next Q ω) - natQueueLyapunov Q) := by
          apply Finset.sum_congr rfl
          intro ω _
          ring

/-- The weighted expected drift is bounded by weighted second-order and
pressure terms. -/
theorem expectedOneStepLyapunovDrift_le_quadratic_pressure
    (Q : NatQueueState I) :
    M.expectedOneStepLyapunovDrift Q
      ≤ ∑ ω : Ω, M.prob Q ω *
          (secondOrderTerm (natQueueToServiceVec (M.arrivals Q ω))
              (natQueueToServiceVec (M.service Q ω))
            + dot (natQueueToServiceVec Q)
                (natQueueToServiceVec (M.arrivals Q ω))
            - dot (natQueueToServiceVec Q)
                (natQueueToServiceVec (M.service Q ω))) := by
  unfold expectedOneStepLyapunovDrift sampleLyapunovDrift next
  apply Finset.sum_le_sum
  intro ω _
  exact mul_le_mul_of_nonneg_left
    (natQueueStep_lyapunov_bound Q (M.arrivals Q ω) (M.service Q ω))
    (M.prob_nonneg Q ω)

/-- Weighted arrival pressure equals queue pressure against the conditional
mean arrival vector. -/
theorem weighted_arrival_pressure_eq (Q : NatQueueState I) :
    (∑ ω : Ω, M.prob Q ω *
        dot (natQueueToServiceVec Q) (natQueueToServiceVec (M.arrivals Q ω)))
      = dot (natQueueToServiceVec Q) (M.expectedArrival Q) := by
  unfold dot expectedArrival natQueueToServiceVec
  calc
    (∑ ω : Ω, M.prob Q ω *
        ∑ i : I, (Q i : ℝ) * (M.arrivals Q ω i : ℝ))
        = ∑ ω : Ω, ∑ i : I,
            M.prob Q ω * ((Q i : ℝ) * (M.arrivals Q ω i : ℝ)) := by
          apply Finset.sum_congr rfl
          intro ω _
          rw [Finset.mul_sum]
    _ = ∑ i : I, ∑ ω : Ω,
            M.prob Q ω * ((Q i : ℝ) * (M.arrivals Q ω i : ℝ)) := by
          rw [Finset.sum_comm]
    _ = ∑ i : I, (Q i : ℝ) *
            ∑ ω : Ω, M.prob Q ω * (M.arrivals Q ω i : ℝ) := by
          apply Finset.sum_congr rfl
          intro i _
          rw [Finset.mul_sum]
          apply Finset.sum_congr rfl
          intro ω _
          ring

/-- Weighted service pressure equals queue pressure against the conditional
mean service vector. -/
theorem weighted_service_pressure_eq (Q : NatQueueState I) :
    (∑ ω : Ω, M.prob Q ω *
        dot (natQueueToServiceVec Q) (natQueueToServiceVec (M.service Q ω)))
      = dot (natQueueToServiceVec Q) (M.expectedService Q) := by
  unfold dot expectedService natQueueToServiceVec
  calc
    (∑ ω : Ω, M.prob Q ω *
        ∑ i : I, (Q i : ℝ) * (M.service Q ω i : ℝ))
        = ∑ ω : Ω, ∑ i : I,
            M.prob Q ω * ((Q i : ℝ) * (M.service Q ω i : ℝ)) := by
          apply Finset.sum_congr rfl
          intro ω _
          rw [Finset.mul_sum]
    _ = ∑ i : I, ∑ ω : Ω,
            M.prob Q ω * ((Q i : ℝ) * (M.service Q ω i : ℝ)) := by
          rw [Finset.sum_comm]
    _ = ∑ i : I, (Q i : ℝ) *
            ∑ ω : Ω, M.prob Q ω * (M.service Q ω i : ℝ) := by
          apply Finset.sum_congr rfl
          intro i _
          rw [Finset.mul_sum]
          apply Finset.sum_congr rfl
          intro ω _
          ring

/-- Coordinatewise conditional arrival bounds imply the queue-pressure
arrival bound used by the drift theorem. -/
theorem arrival_pressure_le_of_expectedArrival_le
    (lambda : NatQueueState I → ServiceVec I)
    (harrival_coord : ∀ Q : NatQueueState I, ∀ i : I,
      M.expectedArrival Q i ≤ lambda Q i) :
    ∀ Q : NatQueueState I,
      (∑ ω : Ω, M.prob Q ω *
          dot (natQueueToServiceVec Q)
            (natQueueToServiceVec (M.arrivals Q ω)))
        ≤ dot (natQueueToServiceVec Q) (lambda Q) := by
  intro Q
  rw [M.weighted_arrival_pressure_eq Q]
  exact dot_mono_service (natQueueToServiceVec_nonnegative Q)
    (harrival_coord Q)

/-- Coordinatewise conditional service lower bounds imply the queue-pressure
service bound used by the drift theorem. -/
theorem service_pressure_ge_of_expectedService_ge
    (lower : NatQueueState I → ServiceVec I)
    (hservice_coord : ∀ Q : NatQueueState I, ∀ i : I,
      lower Q i ≤ M.expectedService Q i) :
    ∀ Q : NatQueueState I,
      dot (natQueueToServiceVec Q) (lower Q)
        ≤ ∑ ω : Ω, M.prob Q ω *
            dot (natQueueToServiceVec Q)
              (natQueueToServiceVec (M.service Q ω)) := by
  intro Q
  rw [M.weighted_service_pressure_eq Q]
  exact dot_mono_service (natQueueToServiceVec_nonnegative Q)
    (hservice_coord Q)

/-- A uniform per-sample second-order bound implies the conditional expected
second-order bound. -/
theorem expectedSecondOrder_le_of_sample_bound
    (B : NatQueueState I → ℝ)
    (hB : ∀ Q : NatQueueState I, ∀ ω : Ω,
      secondOrderTerm (natQueueToServiceVec (M.arrivals Q ω))
        (natQueueToServiceVec (M.service Q ω)) ≤ B Q) :
    ∀ Q : NatQueueState I, M.expectedSecondOrder Q ≤ B Q := by
  intro Q
  unfold expectedSecondOrder
  calc
    (∑ ω : Ω, M.prob Q ω *
      secondOrderTerm (natQueueToServiceVec (M.arrivals Q ω))
        (natQueueToServiceVec (M.service Q ω)))
        ≤ ∑ ω : Ω, M.prob Q ω * B Q := by
          apply Finset.sum_le_sum
          intro ω _
          exact mul_le_mul_of_nonneg_left (hB Q ω) (M.prob_nonneg Q ω)
    _ = (∑ ω : Ω, M.prob Q ω) * B Q := by
          rw [Finset.sum_mul]
    _ = B Q := by
          rw [M.prob_sum Q, one_mul]

/-- Coordinatewise bounds on one arrival/service realization imply a bound
on the quadratic second-order term. -/
theorem secondOrderTerm_natQueueToServiceVec_le_of_coord_bounds
    (Arr Serv : NatQueueState I) (Amax Smax : ServiceVec I)
    (hAmax : Nonnegative Amax) (hSmax : Nonnegative Smax)
    (hArr : ∀ i : I, (Arr i : ℝ) ≤ Amax i)
    (hServ : ∀ i : I, (Serv i : ℝ) ≤ Smax i) :
    secondOrderTerm (natQueueToServiceVec Arr) (natQueueToServiceVec Serv)
      ≤ secondOrderTerm Amax Smax := by
  unfold secondOrderTerm natQueueToServiceVec
  apply Finset.sum_le_sum
  intro i _
  have hArr_nonneg : 0 ≤ (Arr i : ℝ) := Nat.cast_nonneg (Arr i)
  have hServ_nonneg : 0 ≤ (Serv i : ℝ) := Nat.cast_nonneg (Serv i)
  have hArr_abs_to : |(Arr i : ℝ)| ≤ Amax i := by
    rw [abs_of_nonneg hArr_nonneg]
    exact hArr i
  have hServ_abs_to : |(Serv i : ℝ)| ≤ Smax i := by
    rw [abs_of_nonneg hServ_nonneg]
    exact hServ i
  have hArr_abs : |(Arr i : ℝ)| ≤ |Amax i| := by
    simpa [abs_of_nonneg (hAmax i)] using hArr_abs_to
  have hServ_abs : |(Serv i : ℝ)| ≤ |Smax i| := by
    simpa [abs_of_nonneg (hSmax i)] using hServ_abs_to
  have hArr_sq : ((Arr i : ℝ))^2 ≤ (Amax i)^2 := (sq_le_sq).2 hArr_abs
  have hServ_sq : ((Serv i : ℝ))^2 ≤ (Smax i)^2 := (sq_le_sq).2 hServ_abs
  nlinarith

/-- Uniform coordinatewise sample bounds imply a uniform conditional
second-order bound. -/
theorem expectedSecondOrder_le_of_coord_sample_bounds
    (Amax Smax : ServiceVec I)
    (hAmax : Nonnegative Amax) (hSmax : Nonnegative Smax)
    (hArr : ∀ Q : NatQueueState I, ∀ ω : Ω, ∀ i : I,
      (M.arrivals Q ω i : ℝ) ≤ Amax i)
    (hServ : ∀ Q : NatQueueState I, ∀ ω : Ω, ∀ i : I,
      (M.service Q ω i : ℝ) ≤ Smax i) :
    ∀ Q : NatQueueState I,
      M.expectedSecondOrder Q ≤ secondOrderTerm Amax Smax := by
  exact M.expectedSecondOrder_le_of_sample_bound
    (fun _ : NatQueueState I => secondOrderTerm Amax Smax)
    (by
      intro Q ω
      exact secondOrderTerm_natQueueToServiceVec_le_of_coord_bounds
        (M.arrivals Q ω) (M.service Q ω) Amax Smax
        hAmax hSmax (hArr Q ω) (hServ Q ω))

/-- A concrete conditional-drift domination theorem.  If the model has:

* bounded conditional second-order term `B Q`;
* arrival pressure dominated by `lambda Q`;
* service pressure at least `lower Q`;

then its conditional Lyapunov drift is dominated by the deterministic drift
`B Q + Q⋅lambda(Q) - Q⋅lower(Q)`.
-/
theorem expectedLyapunovDrift_le_deterministic_bound
    (B : NatQueueState I → ℝ)
    (lambda lower : NatQueueState I → ServiceVec I)
    (hsecond : ∀ Q : NatQueueState I,
      M.expectedSecondOrder Q ≤ B Q)
    (harrival : ∀ Q : NatQueueState I,
      (∑ ω : Ω, M.prob Q ω *
          dot (natQueueToServiceVec Q)
            (natQueueToServiceVec (M.arrivals Q ω)))
        ≤ dot (natQueueToServiceVec Q) (lambda Q))
    (hservice : ∀ Q : NatQueueState I,
      dot (natQueueToServiceVec Q) (lower Q)
        ≤ ∑ ω : Ω, M.prob Q ω *
            dot (natQueueToServiceVec Q)
              (natQueueToServiceVec (M.service Q ω))) :
    ∀ Q : NatQueueState I,
      expectedLyapunovDrift M.K (natQueueLyapunov (I := I)) Q
        ≤ B Q
          + dot (natQueueToServiceVec Q) (lambda Q)
          - dot (natQueueToServiceVec Q) (lower Q) := by
  intro Q
  have heq := M.expectedLyapunovDrift_eq Q
  have hquad := M.expectedOneStepLyapunovDrift_le_quadratic_pressure Q
  let secondω : Ω → ℝ := fun ω =>
    secondOrderTerm (natQueueToServiceVec (M.arrivals Q ω))
      (natQueueToServiceVec (M.service Q ω))
  let arrivalω : Ω → ℝ := fun ω =>
    dot (natQueueToServiceVec Q) (natQueueToServiceVec (M.arrivals Q ω))
  let serviceω : Ω → ℝ := fun ω =>
    dot (natQueueToServiceVec Q) (natQueueToServiceVec (M.service Q ω))
  have hsplit :
      (∑ ω : Ω, M.prob Q ω *
          (secondω ω + arrivalω ω - serviceω ω))
        = (∑ ω : Ω, M.prob Q ω * secondω ω)
          + (∑ ω : Ω, M.prob Q ω * arrivalω ω)
          - (∑ ω : Ω, M.prob Q ω * serviceω ω) := by
    calc
      (∑ ω : Ω, M.prob Q ω *
          (secondω ω + arrivalω ω - serviceω ω))
          = ∑ ω : Ω,
              (M.prob Q ω * secondω ω
                + M.prob Q ω * arrivalω ω
                - M.prob Q ω * serviceω ω) := by
            apply Finset.sum_congr rfl
            intro ω _
            ring
      _ = (∑ ω : Ω,
              (M.prob Q ω * secondω ω
                + M.prob Q ω * arrivalω ω))
            - ∑ ω : Ω, M.prob Q ω * serviceω ω := by
            rw [Finset.sum_sub_distrib]
      _ = ((∑ ω : Ω, M.prob Q ω * secondω ω)
              + ∑ ω : Ω, M.prob Q ω * arrivalω ω)
            - ∑ ω : Ω, M.prob Q ω * serviceω ω := by
            rw [Finset.sum_add_distrib]
      _ = (∑ ω : Ω, M.prob Q ω * secondω ω)
          + (∑ ω : Ω, M.prob Q ω * arrivalω ω)
          - (∑ ω : Ω, M.prob Q ω * serviceω ω) := by
            ring
  have hsecondQ := hsecond Q
  have harrivalQ := harrival Q
  have hserviceQ := hservice Q
  unfold secondω arrivalω serviceω at hsplit
  rw [heq]
  calc
    M.expectedOneStepLyapunovDrift Q
        ≤ ∑ ω : Ω, M.prob Q ω *
            (secondOrderTerm (natQueueToServiceVec (M.arrivals Q ω))
                (natQueueToServiceVec (M.service Q ω))
              + dot (natQueueToServiceVec Q)
                  (natQueueToServiceVec (M.arrivals Q ω))
              - dot (natQueueToServiceVec Q)
                  (natQueueToServiceVec (M.service Q ω))) := hquad
    _ = (∑ ω : Ω, M.prob Q ω *
            secondOrderTerm (natQueueToServiceVec (M.arrivals Q ω))
              (natQueueToServiceVec (M.service Q ω)))
          + (∑ ω : Ω, M.prob Q ω *
            dot (natQueueToServiceVec Q)
              (natQueueToServiceVec (M.arrivals Q ω)))
          - (∑ ω : Ω, M.prob Q ω *
            dot (natQueueToServiceVec Q)
              (natQueueToServiceVec (M.service Q ω))) := hsplit
    _ ≤ B Q
          + dot (natQueueToServiceVec Q) (lambda Q)
          - dot (natQueueToServiceVec Q) (lower Q) := by
        unfold expectedSecondOrder at hsecondQ
        linarith

/-- The finite-support stochastic model packaged as a `NatQueueTransitionModel`
with the `drift_dominated` field proved from explicit moment and pressure
assumptions. -/
def toNatQueueTransitionModel
    (B : NatQueueState I → ℝ)
    (lambda lower : NatQueueState I → ServiceVec I)
    (hsecond : ∀ Q : NatQueueState I,
      M.expectedSecondOrder Q ≤ B Q)
    (harrival : ∀ Q : NatQueueState I,
      (∑ ω : Ω, M.prob Q ω *
          dot (natQueueToServiceVec Q)
            (natQueueToServiceVec (M.arrivals Q ω)))
        ≤ dot (natQueueToServiceVec Q) (lambda Q))
    (hservice : ∀ Q : NatQueueState I,
      dot (natQueueToServiceVec Q) (lower Q)
        ≤ ∑ ω : Ω, M.prob Q ω *
            dot (natQueueToServiceVec Q)
              (natQueueToServiceVec (M.service Q ω))) :
    NatQueueTransitionModel I where
  K := M.K
  deterministicDrift := fun Q =>
    B Q + dot (natQueueToServiceVec Q) (lambda Q)
      - dot (natQueueToServiceVec Q) (lower Q)
  drift_dominated :=
    M.expectedLyapunovDrift_le_deterministic_bound
      B lambda lower hsecond harrival hservice

/-- Concrete finite-support stochastic stability theorem: once the explicit
finite model proves conditional drift domination and the resulting
deterministic bound is linearly negative outside large queues, the integer
queue is positive recurrent through a finite backlog set. -/
theorem positive_recurrent_via_finite_set
    (B : NatQueueState I → ℝ)
    (lambda lower : NatQueueState I → ServiceVec I)
    (hsecond : ∀ Q : NatQueueState I,
      M.expectedSecondOrder Q ≤ B Q)
    (harrival : ∀ Q : NatQueueState I,
      (∑ ω : Ω, M.prob Q ω *
          dot (natQueueToServiceVec Q)
            (natQueueToServiceVec (M.arrivals Q ω)))
        ≤ dot (natQueueToServiceVec Q) (lambda Q))
    (hservice : ∀ Q : NatQueueState I,
      dot (natQueueToServiceVec Q) (lower Q)
        ≤ ∑ ω : Ω, M.prob Q ω *
            dot (natQueueToServiceVec Q)
              (natQueueToServiceVec (M.service Q ω)))
    {B0 η α : ℝ} {N : ℕ}
    (hη : 0 < η) (hα : 0 < α)
    (hN : B0 + α ≤ η * (N : ℝ))
    (hlinear : ∀ Q : NatQueueState I,
      B Q + dot (natQueueToServiceVec Q) (lambda Q)
          - dot (natQueueToServiceVec Q) (lower Q)
        ≤ B0 - η * natQueueL1 Q)
    (hreturn : ∀ Q : NatQueueState I,
      natQueueSmallSet (I := I) N Q →
        FiniteExpectedReturnTimeToSet M.K (natQueueSmallSet (I := I) N) Q) :
    PositiveRecurrentViaFiniteSet M.K (natQueueSmallSet (I := I) N) := by
  let TM := M.toNatQueueTransitionModel B lambda lower hsecond harrival hservice
  exact nat_model_linear_drift_positive_recurrent_via_finite_set
    TM hη hα hN hlinear hreturn

/-- Concrete finite-support stochastic stability theorem with local return
discharged by the Foster-Lyapunov drift certificate. -/
theorem positive_recurrent_via_finite_set'
    (B : NatQueueState I → ℝ)
    (lambda lower : NatQueueState I → ServiceVec I)
    (hsecond : ∀ Q : NatQueueState I,
      M.expectedSecondOrder Q ≤ B Q)
    (harrival : ∀ Q : NatQueueState I,
      (∑ ω : Ω, M.prob Q ω *
          dot (natQueueToServiceVec Q)
            (natQueueToServiceVec (M.arrivals Q ω)))
        ≤ dot (natQueueToServiceVec Q) (lambda Q))
    (hservice : ∀ Q : NatQueueState I,
      dot (natQueueToServiceVec Q) (lower Q)
        ≤ ∑ ω : Ω, M.prob Q ω *
            dot (natQueueToServiceVec Q)
              (natQueueToServiceVec (M.service Q ω)))
    {B0 η α : ℝ} {N : ℕ}
    (hη : 0 < η) (hα : 0 < α)
    (hN : B0 + α ≤ η * (N : ℝ))
    (hlinear : ∀ Q : NatQueueState I,
      B Q + dot (natQueueToServiceVec Q) (lambda Q)
          - dot (natQueueToServiceVec Q) (lower Q)
        ≤ B0 - η * natQueueL1 Q) :
    PositiveRecurrentViaFiniteSet M.K (natQueueSmallSet (I := I) N) := by
  let TM := M.toNatQueueTransitionModel B lambda lower hsecond harrival hservice
  exact nat_model_linear_drift_positive_recurrent_via_finite_set'
    TM hη hα hN hlinear

/-- Concrete finite-support stochastic stability theorem from primitive
coordinate moment conditions.  This removes the aggregate arrival/service
pressure assumptions: they are proved from conditional coordinate means. -/
theorem positive_recurrent_via_coordinate_moments
    (B : NatQueueState I → ℝ)
    (lambda lower : NatQueueState I → ServiceVec I)
    (hsecond : ∀ Q : NatQueueState I, M.expectedSecondOrder Q ≤ B Q)
    (harrival_coord : ∀ Q : NatQueueState I, ∀ i : I,
      M.expectedArrival Q i ≤ lambda Q i)
    (hservice_coord : ∀ Q : NatQueueState I, ∀ i : I,
      lower Q i ≤ M.expectedService Q i)
    {B0 η α : ℝ} {N : ℕ}
    (hη : 0 < η) (hα : 0 < α)
    (hN : B0 + α ≤ η * (N : ℝ))
    (hlinear : ∀ Q : NatQueueState I,
      B Q + dot (natQueueToServiceVec Q) (lambda Q)
          - dot (natQueueToServiceVec Q) (lower Q)
        ≤ B0 - η * natQueueL1 Q)
    (hreturn : ∀ Q : NatQueueState I,
      natQueueSmallSet (I := I) N Q →
        FiniteExpectedReturnTimeToSet M.K (natQueueSmallSet (I := I) N) Q) :
    PositiveRecurrentViaFiniteSet M.K (natQueueSmallSet (I := I) N) := by
  exact M.positive_recurrent_via_finite_set
    B lambda lower hsecond
    (M.arrival_pressure_le_of_expectedArrival_le lambda harrival_coord)
    (M.service_pressure_ge_of_expectedService_ge lower hservice_coord)
    hη hα hN hlinear hreturn

/-- Coordinate-moment concrete stochastic stability with the finite-small-set
local return bound proved from Foster drift. -/
theorem positive_recurrent_via_coordinate_moments'
    (B : NatQueueState I → ℝ)
    (lambda lower : NatQueueState I → ServiceVec I)
    (hsecond : ∀ Q : NatQueueState I, M.expectedSecondOrder Q ≤ B Q)
    (harrival_coord : ∀ Q : NatQueueState I, ∀ i : I,
      M.expectedArrival Q i ≤ lambda Q i)
    (hservice_coord : ∀ Q : NatQueueState I, ∀ i : I,
      lower Q i ≤ M.expectedService Q i)
    {B0 η α : ℝ} {N : ℕ}
    (hη : 0 < η) (hα : 0 < α)
    (hN : B0 + α ≤ η * (N : ℝ))
    (hlinear : ∀ Q : NatQueueState I,
      B Q + dot (natQueueToServiceVec Q) (lambda Q)
          - dot (natQueueToServiceVec Q) (lower Q)
        ≤ B0 - η * natQueueL1 Q) :
    PositiveRecurrentViaFiniteSet M.K (natQueueSmallSet (I := I) N) := by
  exact M.positive_recurrent_via_finite_set'
    B lambda lower hsecond
    (M.arrival_pressure_le_of_expectedArrival_le lambda harrival_coord)
    (M.service_pressure_ge_of_expectedService_ge lower hservice_coord)
    hη hα hN hlinear

end FiniteSupportQueueModel

end Scheduleurm


/-! ## Source: Scheduleurm/OperationalCapacity.lean -/


/-!
# Scheduleurm: operational capacity-region semantics

`InCapacityWithSlack` is a convex-geometric statement.  A paper-level
capacity theorem needs an operational stability semantics.  This file defines
that semantics using the integer-queue recurrence layer and states the exact
soundness/necessity boundary:

* soundness: a verified drift-dominated Markov queue model gives operational
  stability for a model that explicitly certifies the offered load;
* characterization: positive geometric slack is sufficient for operational
  stabilizability, while the model-specific conservation law gives the
  zero-slack capacity-closure necessity direction.
-/

noncomputable section

set_option linter.unusedSectionVars false
set_option linter.unusedVariables false

namespace Scheduleurm

variable {I A : Type*} [Fintype I] [DecidableEq I] [DecidableEq A]
variable {Ω : Type} [Fintype Ω]

/-- Operational stability for a concrete integer-valued queue Markov model:
the model is positive recurrent through some finite backlog sublevel set. -/
def OperationallyStableIntegerQueueModel
    (M : NatQueueTransitionModel I) : Prop :=
  ∃ N : ℕ, PositiveRecurrentViaFiniteSet M.K (natQueueSmallSet (I := I) N)

/-- Certificate that a concrete integer queue transition model is actually a
model of the offered load `lam`.

The certificate is deliberately stronger than a bare predicate on an abstract
transition kernel.  It exposes a finite-support arrival/service realization
model, proves that this concrete model induces the same transition expectation
as `M`, and identifies the model's conditional mean arrival vector with the
load vector `lam`.  The service accounting vector is also tied to the concrete
model's conditional mean service.

This is the formal guard against the vacuous statement "some stable Markov
chain exists" for an unrelated `lam`. -/
structure ModelEncodesLoad
    (M : NatQueueTransitionModel I) (lam : ServiceVec I) where
  Ω : Type
  instΩ : Fintype Ω
  finiteModel : @FiniteSupportQueueModel I Ω _ instΩ
  transition_eq : M.K = finiteModel.K
  load_nonnegative : Nonnegative lam
  arrival_mean_eq_load : ∀ Q : NatQueueState I, ∀ i : I,
    finiteModel.expectedArrival Q i = lam i
  serviceMean : NatQueueState I → ServiceVec I
  service_mean_eq_expected : ∀ Q : NatQueueState I, ∀ i : I,
    serviceMean Q i = finiteModel.expectedService Q i

/-- A queue model bundled with the certificate that it is a model of `lam`. -/
structure LoadCertifiedNatQueueModel
    (lam : ServiceVec I) where
  model : NatQueueTransitionModel I
  encodes_load : ModelEncodesLoad model lam

/-- Constructor for the common case where the transition model is already
known to have the same kernel as an explicit finite-support arrival/service
model and that model's conditional mean arrivals are exactly `lam`. -/
def finite_support_model_encodes_load
    (M : NatQueueTransitionModel I) (FM : FiniteSupportQueueModel I Ω)
    (lam : ServiceVec I)
    (hK : M.K = FM.K)
    (hload : Nonnegative lam)
    (harrival_eq : ∀ Q : NatQueueState I, ∀ i : I,
      FM.expectedArrival Q i = lam i) :
    ModelEncodesLoad M lam := by
  exact
    { Ω := Ω
      instΩ := inferInstance
      finiteModel := FM
      transition_eq := hK
      load_nonnegative := hload
      arrival_mean_eq_load := harrival_eq
      serviceMean := FM.expectedService
      service_mean_eq_expected := by
        intro Q i
        rfl }

/-- A load vector is operationally stabilizable if some verified integer-queue
Markov model that explicitly encodes that same load stabilizes it. -/
def OperationallyStabilizesIntegerLoad
    (lam : ServiceVec I) : Prop :=
  ∃ C : LoadCertifiedNatQueueModel (I := I) lam,
    OperationallyStableIntegerQueueModel C.model

/-- A verified integer-queue drift model is an operational stabilizer. -/
theorem integer_model_operationally_stable_from_verified_drift
    (M : NatQueueTransitionModel I)
    {B η α : ℝ} {N : ℕ}
    (hη : 0 < η) (hα : 0 < α)
    (hN : B + α ≤ η * (N : ℝ))
    (hdet : ∀ Q : NatQueueState I,
      M.deterministicDrift Q ≤ B - η * natQueueL1 Q)
    (hreturn : ∀ Q : NatQueueState I,
      natQueueSmallSet (I := I) N Q →
        FiniteExpectedReturnTimeToSet M.K (natQueueSmallSet (I := I) N) Q) :
    OperationallyStableIntegerQueueModel M := by
  refine ⟨N, ?_⟩
  exact nat_model_linear_drift_positive_recurrent_via_finite_set
    M hη hα hN hdet hreturn

/-- Capacity-region soundness under a verified stochastic queue model.

The geometric capacity condition is kept in the statement because it is the
paper-level premise.  The actual operational work is the model-specific
verified drift bound `hdet`, which should be derived from the MaxWeight
capacity slack theorem for the concrete stochastic scheduler. -/
theorem capacity_region_operational_sound_under_verified_drift
    (F : ActionFamily A) (μ : A → ServiceVec I)
    {lam : ServiceVec I} {δ B η α : ℝ} {N : ℕ}
    (hcap : InCapacityWithSlack F μ lam δ)
    (M : NatQueueTransitionModel I)
    (hencode : ModelEncodesLoad M lam)
    (hη : 0 < η) (hα : 0 < α)
    (hN : B + α ≤ η * (N : ℝ))
    (hdet : ∀ Q : NatQueueState I,
      M.deterministicDrift Q ≤ B - η * natQueueL1 Q)
    (hreturn : ∀ Q : NatQueueState I,
      natQueueSmallSet (I := I) N Q →
        FiniteExpectedReturnTimeToSet M.K (natQueueSmallSet (I := I) N) Q) :
    OperationallyStabilizesIntegerLoad lam := by
  exact ⟨⟨M, hencode⟩, integer_model_operationally_stable_from_verified_drift
    M hη hα hN hdet hreturn⟩

/-- Soundness predicate: every load in the geometric capacity region has a
verified operational stabilizer. -/
def CapacityRegionOperationalSound
    (F : ActionFamily A) (μ : A → ServiceVec I) : Prop :=
  ∀ lam δ, InCapacityWithSlack F μ lam δ →
    OperationallyStabilizesIntegerLoad (I := I) lam

/-- A concrete conservation-law certificate: the long-run occupation measure
over scheduler actions is a stationary mix, and its average service dominates
the average arrival vector.  This is the part that must be derived from the
chosen stochastic arrival/service model and workload accounting. -/
structure OperationalConservationCertificate
    (F : ActionFamily A) (μ : A → ServiceVec I) (lam : ServiceVec I) where
  occupation : StationaryMix F
  arrival_le_service :
    ∀ i : I, lam i ≤ mixedService F μ occupation i

/-- A conservation-law certificate places the load in the zero-slack
capacity region. -/
theorem conservation_certificate_implies_capacity_closure
    (F : ActionFamily A) (μ : A → ServiceVec I) (lam : ServiceVec I)
    (cert : OperationalConservationCertificate F μ lam) :
    InCapacityWithSlack F μ lam 0 := by
  refine ⟨cert.occupation, ?_⟩
  intro i
  simpa using cert.arrival_le_service i

/-- Model-specific conservation-law predicate: every operationally stable
integer-load model admits a stationary occupation certificate. -/
def CapacityRegionOperationalConservationLaw
    (F : ActionFamily A) (μ : A → ServiceVec I) : Prop :=
  ∀ lam, OperationallyStabilizesIntegerLoad (I := I) lam →
    Nonempty (OperationalConservationCertificate F μ lam)

/-- Necessity predicate: any operationally stabilizable load must lie in the
zero-slack geometric capacity closure.  This is the conservation-law direction
and must be proved from the concrete service/arrival model. -/
def CapacityRegionOperationalNecessary
    (F : ActionFamily A) (μ : A → ServiceVec I) : Prop :=
  ∀ lam, OperationallyStabilizesIntegerLoad (I := I) lam →
    InCapacityWithSlack F μ lam 0

/-- A model-specific conservation law is exactly the missing necessity
bridge from operational stability to capacity-closure membership. -/
theorem conservation_law_implies_operational_necessity
    (F : ActionFamily A) (μ : A → ServiceVec I)
    (hcons : CapacityRegionOperationalConservationLaw F μ) :
    CapacityRegionOperationalNecessary F μ := by
  intro lam hstable
  rcases hcons lam hstable with ⟨cert⟩
  exact conservation_certificate_implies_capacity_closure
    F μ lam cert

/-- Operational capacity-region sandwich once both model-specific directions
have been verified: positive slack is sufficient for stability, and any
stabilizable load must be in the zero-slack capacity closure. -/
theorem configuration_capacity_region_operational_sandwich
    (F : ActionFamily A) (μ : A → ServiceVec I)
    (hsound : CapacityRegionOperationalSound F μ)
    (hnecessary : CapacityRegionOperationalNecessary F μ) :
    (∀ lam δ,
      InCapacityWithSlack F μ lam δ →
        OperationallyStabilizesIntegerLoad (I := I) lam) ∧
      (∀ lam,
        OperationallyStabilizesIntegerLoad (I := I) lam →
          InCapacityWithSlack F μ lam 0) := by
  constructor
  · intro lam δ hcap
    exact hsound lam δ hcap
  · exact hnecessary

/-- If the model proves the stronger fact that every stabilizable load has a
specified slack `δ`, then the usual iff statement at that slack follows.  This
is intentionally separated from the conservation-law closure theorem because
positive slack is not a consequence of stability alone. -/
theorem configuration_capacity_region_operational_characterization_at_slack
    (F : ActionFamily A) (μ : A → ServiceVec I) (δ : ℝ)
    (hsound : CapacityRegionOperationalSound F μ)
    (hnecessary_at_slack :
      ∀ lam, OperationallyStabilizesIntegerLoad (I := I) lam →
        InCapacityWithSlack F μ lam δ) :
    ∀ lam,
      InCapacityWithSlack F μ lam δ ↔
        OperationallyStabilizesIntegerLoad (I := I) lam := by
  intro lam
  constructor
  · exact hsound lam δ
  · exact hnecessary_at_slack lam

end Scheduleurm


/-! ## Source: Scheduleurm/RegimeStability.lean -/


/-!
# Scheduleurm: uniform and average hidden-regime stability

Hidden regimes create two different stability claims:

* **uniform-in-regime**: the arrival rate has slack in every regime;
* **average-regime**: the arrival rate has slack only for a regime mixture.

They are not interchangeable.  This file states both objects explicitly and
proves their support-function consequences.
-/

noncomputable section

set_option linter.unusedSectionVars false
set_option linter.unusedVariables false

namespace Scheduleurm

variable {I A Z : Type*} [Fintype I] [DecidableEq A] [Fintype Z]

/-- Uniform capacity slack in every hidden regime. -/
def UniformRegimeCapacityWithSlack
    (F : ActionFamily A) (μ : Z → A → ServiceVec I)
    (lam : ServiceVec I) (δ : ℝ) : Prop :=
  ∀ z : Z, InCapacityWithSlack F (μ z) lam δ

/-- Average-regime capacity slack under a regime belief / mixture `β`. -/
def AverageRegimeCapacityWithSlack
    (F : ActionFamily A) (β : Z → ℝ) (μ : Z → A → ServiceVec I)
    (lam : ServiceVec I) (δ : ℝ) : Prop :=
  InCapacityWithSlack F (beliefWeightedService β μ) lam δ

/-- Uniform regime slack gives the usual MaxWeight support slack in whichever
regime is active. -/
theorem uniform_regime_capacity_support_slack
    (F : ActionFamily A) (μ : Z → A → ServiceVec I)
    {lam q : ServiceVec I} {δ : ℝ}
    (hq : Nonnegative q)
    (hcap : UniformRegimeCapacityWithSlack F μ lam δ) :
    ∀ z : Z, dot q lam + δ * l1 q ≤ support F (μ z) q := by
  intro z
  exact capacity_slack_implies_support_slack F (μ z) hq (hcap z)

/-- Average-regime slack gives support slack only for the belief-weighted
service model. -/
theorem average_regime_capacity_support_slack
    (F : ActionFamily A) (β : Z → ℝ) (μ : Z → A → ServiceVec I)
    {lam q : ServiceVec I} {δ : ℝ}
    (hq : Nonnegative q)
    (hcap : AverageRegimeCapacityWithSlack F β μ lam δ) :
    dot q lam + δ * l1 q
      ≤ support F (beliefWeightedService β μ) q := by
  exact capacity_slack_implies_support_slack F
    (beliefWeightedService β μ) hq hcap

/-- Uniform slack implies average-regime slack if the same stationary mix is
available uniformly.  This stronger assumption is useful when the scheduler
must be stable in every segment, not just on a long-run average. -/
theorem common_mix_uniform_implies_average_capacity
    (F : ActionFamily A) (β : Z → ℝ) (μ : Z → A → ServiceVec I)
    {lam : ServiceVec I} {δ : ℝ}
    (hβ : RegimeBeliefValid β)
    (x : StationaryMix F)
    (hx : ∀ z i, lam i + δ ≤ mixedService F (μ z) x i) :
    AverageRegimeCapacityWithSlack F β μ lam δ := by
  refine ⟨x, ?_⟩
  intro i
  rcases hβ with ⟨hβ_nonneg, hβ_sum⟩
  calc
    lam i + δ
        = (∑ z : Z, β z) * (lam i + δ) := by
          rw [hβ_sum]
          ring
    _ = ∑ z : Z, β z * (lam i + δ) := by
          rw [Finset.sum_mul]
    _ ≤ ∑ z : Z, β z * mixedService F (μ z) x i := by
          apply Finset.sum_le_sum
          intro z _
          exact mul_le_mul_of_nonneg_left (hx z i) (hβ_nonneg z)
    _ = mixedService F (beliefWeightedService β μ) x i := by
          unfold mixedService beliefWeightedService
          calc
            ∑ z : Z, β z * ∑ a ∈ F.acts, x.weight a * μ z a i
                = ∑ z : Z, ∑ a ∈ F.acts,
                    β z * (x.weight a * μ z a i) := by
                  apply Finset.sum_congr rfl
                  intro z _
                  rw [Finset.mul_sum]
            _ = ∑ a ∈ F.acts, ∑ z : Z,
                    β z * (x.weight a * μ z a i) := by
                  rw [Finset.sum_comm]
            _ = ∑ a ∈ F.acts, x.weight a * ∑ z : Z,
                    β z * μ z a i := by
                  apply Finset.sum_congr rfl
                  intro a _
                  rw [Finset.mul_sum]
                  apply Finset.sum_congr rfl
                  intro z _
                  ring

end Scheduleurm


/-! ## Source: Scheduleurm/Concentration.lean -/


/-!
# Scheduleurm: confidence-radius concentration envelopes

This file formalises the deterministic concentration shell used by the
unknown-service theorem.

The probabilistic concentration lemma for the observations should provide
bucket-level cumulative radius bounds of the form

`loss_b ≤ C sqrt(N_b)`.

The proofs here show how those local bounds aggregate into the familiar
`C sqrt(|A||Z|T)` term by Cauchy-Schwarz, and how that envelope feeds into
the LCB support-regret theorem.
-/

noncomputable section

set_option linter.unusedSectionVars false
set_option linter.unusedVariables false

namespace Scheduleurm

open BigOperators

variable {I A T Bkt : Type*}
  [Fintype I] [DecidableEq A] [Fintype T] [Fintype Bkt]

/-- Cauchy-Schwarz form used to sum per-action/per-regime confidence
radii. -/
theorem sum_sqrt_le_sqrt_card_mul_sum
    (count : Bkt → ℝ)
    (hcount : ∀ b : Bkt, 0 ≤ count b) :
    (∑ b : Bkt, Real.sqrt (count b))
      ≤ Real.sqrt ((Fintype.card Bkt : ℝ) * (∑ b : Bkt, count b)) := by
  have h := Real.sum_sqrt_mul_sqrt_le (Finset.univ : Finset Bkt)
    (fun _ : Bkt => by positivity : ∀ b : Bkt, 0 ≤ (1 : ℝ)) hcount
  simpa using h

/-- Bucket-level confidence-radius bounds aggregate to a
`sqrt(number-of-buckets × horizon-mass)` envelope.  Instantiate `Bkt` as
`A × Z` for the `sqrt(|A||Z|T)` term. -/
theorem cumulative_radius_from_bucket_sqrt_bounds
    (radius : T → ℝ) (bucketLoss bucketCount : Bkt → ℝ)
    {C horizonMass : ℝ}
    (hC : 0 ≤ C)
    (htotal : (∑ t : T, radius t) ≤ ∑ b : Bkt, bucketLoss b)
    (hbucket : ∀ b : Bkt,
      bucketLoss b ≤ C * Real.sqrt (bucketCount b))
    (hcount : ∀ b : Bkt, 0 ≤ bucketCount b)
    (hcount_sum : (∑ b : Bkt, bucketCount b) = horizonMass) :
    (∑ t : T, radius t)
      ≤ C * Real.sqrt ((Fintype.card Bkt : ℝ) * horizonMass) := by
  have hloss :
      (∑ b : Bkt, bucketLoss b)
        ≤ ∑ b : Bkt, C * Real.sqrt (bucketCount b) := by
    exact Finset.sum_le_sum (fun b _ => hbucket b)
  have hfactor :
      (∑ b : Bkt, C * Real.sqrt (bucketCount b))
        = C * ∑ b : Bkt, Real.sqrt (bucketCount b) := by
    rw [Finset.mul_sum]
  have hsqrt :
      (∑ b : Bkt, Real.sqrt (bucketCount b))
        ≤ Real.sqrt ((Fintype.card Bkt : ℝ) * horizonMass) := by
    simpa [hcount_sum] using sum_sqrt_le_sqrt_card_mul_sum
      (Bkt := Bkt) bucketCount hcount
  have hmul :
      C * (∑ b : Bkt, Real.sqrt (bucketCount b))
        ≤ C * Real.sqrt ((Fintype.card Bkt : ℝ) * horizonMass) :=
    mul_le_mul_of_nonneg_left hsqrt hC
  linarith

/-- If backlogs are bounded by `Qmax` on the horizon, the cumulative LCB
learning support regret inherits the `sqrt(|A||Z|T)` confidence envelope. -/
theorem cumulative_lcb_learning_regret_sqrt_envelope
    (F : ActionFamily A)
    (μtrue μhat : T → A → ServiceVec I)
    (rad : T → A → ℝ)
    (Q : T → ServiceVec I) (ε : T → ℝ)
    {Qmax C horizonMass : ℝ}
    (hQ : ∀ t : T, Nonnegative (Q t))
    (hε : ∀ t : T, 0 ≤ ε t)
    (hconf : ∀ t : T,
      ServiceConfidenceEvent F (μtrue t) (μhat t) (rad t))
    (hrad : ∀ t : T, RadiusBounded F (rad t) (ε t))
    (hQmax : ∀ t : T, l1 (Q t) ≤ Qmax)
    (hCenv : (∑ t : T, ε t)
      ≤ C * Real.sqrt ((Fintype.card Bkt : ℝ) * horizonMass))
    (hQmax_nonneg : 0 ≤ Qmax) :
    (∑ t : T,
      (support F (μtrue t) (Q t)
        - support F (lcbService (μhat t) (rad t)) (Q t)))
      ≤
    2 * Qmax * C * Real.sqrt ((Fintype.card Bkt : ℝ) * horizonMass) := by
  have hbase := cumulative_lcb_learning_support_regret
    F μtrue μhat rad Q ε hQ hε hconf hrad
  have hterm :
      (∑ t : T, (2 * ε t) * l1 (Q t))
        ≤ ∑ t : T, (2 * ε t) * Qmax := by
    apply Finset.sum_le_sum
    intro t _
    exact mul_le_mul_of_nonneg_left (hQmax t) (by nlinarith [hε t])
  have hfactor :
      (∑ t : T, (2 * ε t) * Qmax)
        = 2 * Qmax * (∑ t : T, ε t) := by
    calc
      (∑ t : T, (2 * ε t) * Qmax)
          = ∑ t : T, (2 * Qmax) * ε t := by
            apply Finset.sum_congr rfl
            intro t _
            ring
      _ = (2 * Qmax) * (∑ t : T, ε t) := by
            rw [Finset.mul_sum]
      _ = 2 * Qmax * (∑ t : T, ε t) := by
            ring
  have hmul :
      2 * Qmax * (∑ t : T, ε t)
        ≤ 2 * Qmax
            * (C * Real.sqrt ((Fintype.card Bkt : ℝ) * horizonMass)) := by
    exact mul_le_mul_of_nonneg_left hCenv (by nlinarith)
  have htarget :
      2 * Qmax
            * (C * Real.sqrt ((Fintype.card Bkt : ℝ) * horizonMass))
        = 2 * Qmax * C
            * Real.sqrt ((Fintype.card Bkt : ℝ) * horizonMass) := by
    ring
  linarith

end Scheduleurm


/-! ## Source: Scheduleurm/ConcentrationProbability.lean -/


/-!
# Scheduleurm: concentration-event interface

This file adds the probability-facing boundary for Theorem D.  It does not
pretend to prove a Hoeffding/UCB theorem without a probability space; instead
it defines the exact high-probability event that such a theorem must deliver
and proves that, on that event, the deterministic `sqrt(|A||Z|T)` learning
regret bound follows.
-/

noncomputable section

set_option linter.unusedSectionVars false
set_option linter.unusedVariables false

namespace Scheduleurm

open BigOperators

variable {I A T Bkt : Type*}
  [Fintype I] [DecidableEq A] [Fintype T] [Fintype Bkt]

/-- Bucket-level concentration event.  A probabilistic theorem should prove
that this event holds with high probability under sub-Gaussian/noise and
adaptive-sampling assumptions. -/
def BucketConcentrationEvent
    (radius : T → ℝ) (bucketLoss bucketCount : Bkt → ℝ)
    (C horizonMass : ℝ) : Prop :=
  0 ≤ C ∧
    (∑ t : T, radius t) ≤ ∑ b : Bkt, bucketLoss b ∧
    (∀ b : Bkt, bucketLoss b ≤ C * Real.sqrt (bucketCount b)) ∧
    (∀ b : Bkt, 0 ≤ bucketCount b) ∧
    (∑ b : Bkt, bucketCount b) = horizonMass

/-- A bucket concentration event gives the cumulative confidence-radius
envelope used by the learning-regret proof. -/
theorem bucket_concentration_event_radius_envelope
    (radius : T → ℝ) (bucketLoss bucketCount : Bkt → ℝ)
    {C horizonMass : ℝ}
    (hevent : BucketConcentrationEvent radius bucketLoss bucketCount C horizonMass) :
    (∑ t : T, radius t)
      ≤ C * Real.sqrt ((Fintype.card Bkt : ℝ) * horizonMass) := by
  rcases hevent with ⟨hC, htotal, hbucket, hcount, hsum⟩
  exact cumulative_radius_from_bucket_sqrt_bounds
    radius bucketLoss bucketCount hC htotal hbucket hcount hsum

/-- LCB confidence event over all slots. -/
def LCBConfidencePathEvent
    (F : ActionFamily A)
    (μtrue μhat : T → A → ServiceVec I)
    (rad : T → A → ℝ) (ε : T → ℝ) : Prop :=
  (∀ t : T, ServiceConfidenceEvent F (μtrue t) (μhat t) (rad t)) ∧
    (∀ t : T, RadiusBounded F (rad t) (ε t))

/-- On the LCB confidence event and bucket concentration event, cumulative
learning regret has the advertised `sqrt(|A||Z|T)` envelope. -/
theorem lcb_learning_regret_from_concentration_events
    (F : ActionFamily A)
    (μtrue μhat : T → A → ServiceVec I)
    (rad : T → A → ℝ)
    (Q : T → ServiceVec I) (ε : T → ℝ)
    (bucketLoss bucketCount : Bkt → ℝ)
    {Qmax C horizonMass : ℝ}
    (hQ : ∀ t : T, Nonnegative (Q t))
    (hε : ∀ t : T, 0 ≤ ε t)
    (hpath : LCBConfidencePathEvent F μtrue μhat rad ε)
    (hbucket : BucketConcentrationEvent ε bucketLoss bucketCount C horizonMass)
    (hQmax : ∀ t : T, l1 (Q t) ≤ Qmax)
    (hQmax_nonneg : 0 ≤ Qmax) :
    (∑ t : T,
      (support F (μtrue t) (Q t)
        - support F (lcbService (μhat t) (rad t)) (Q t)))
      ≤
    2 * Qmax * C * Real.sqrt ((Fintype.card Bkt : ℝ) * horizonMass) := by
  rcases hpath with ⟨hconf, hrad⟩
  have henv :
      (∑ t : T, ε t)
        ≤ C * Real.sqrt ((Fintype.card Bkt : ℝ) * horizonMass) :=
    bucket_concentration_event_radius_envelope
      ε bucketLoss bucketCount hbucket
  exact cumulative_lcb_learning_regret_sqrt_envelope
    (Bkt := Bkt) F μtrue μhat rad Q ε hQ hε hconf hrad
    hQmax henv hQmax_nonneg

end Scheduleurm


/-! ## Source: Scheduleurm/StructuredLearning.lean -/


/-!
# Scheduleurm: structured learning envelopes

A regret bound depending on the full global action count `|A|` is often
vacuous because configuration actions are combinatorial.  This file proves
the same concentration aggregation over an explicit finite active bucket set.
The bucket set may represent local fabric neighborhoods, co-location profiles,
semi-bandit factors, or any lower-dimensional structure supplied by the
system model.
-/

noncomputable section

set_option linter.unusedSectionVars false
set_option linter.unusedVariables false

namespace Scheduleurm

open BigOperators MeasureTheory

variable {Bkt T : Type*} [DecidableEq Bkt] [Fintype T]

/-- Cauchy-Schwarz over an active bucket set, rather than over all global
configuration actions. -/
theorem sum_sqrt_le_sqrt_finset_card_mul_sum
    (active : Finset Bkt) (count : Bkt → ℝ)
    (hcount : ∀ b ∈ active, 0 ≤ count b) :
    (∑ b ∈ active, Real.sqrt (count b))
      ≤ Real.sqrt ((active.card : ℝ) * (∑ b ∈ active, count b)) := by
  let countActive : Bkt → ℝ := fun b => if b ∈ active then count b else 0
  have hnonneg : ∀ b : Bkt, 0 ≤ countActive b := by
    intro b
    by_cases hb : b ∈ active
    · simp [countActive, hb, hcount b hb]
    · simp [countActive, hb]
  have h := Real.sum_sqrt_mul_sqrt_le active
    (f := fun _ : Bkt => (1 : ℝ)) (g := countActive)
    (fun _ => by positivity) hnonneg
  have hleft :
      (∑ b ∈ active, Real.sqrt (countActive b))
        = ∑ b ∈ active, Real.sqrt (count b) := by
    apply Finset.sum_congr rfl
    intro b hb
    simp [countActive, hb]
  have hsum :
      (∑ b ∈ active, countActive b)
        = ∑ b ∈ active, count b := by
    apply Finset.sum_congr rfl
    intro b hb
    simp [countActive, hb]
  have hsum_nonneg : 0 ≤ ∑ b ∈ active, count b :=
    Finset.sum_nonneg (fun b hb => hcount b hb)
  simp at h
  rw [hleft, hsum] at h
  rwa [Real.sqrt_mul (by positivity) (∑ b ∈ active, count b)]

/-- Bucket-level losses aggregate with dependence on the number of active
structured buckets, not the full global configuration count. -/
theorem cumulative_radius_from_active_bucket_sqrt_bounds
    (active : Finset Bkt)
    (radius : T → ℝ) (bucketLoss bucketCount : Bkt → ℝ)
    {C horizonMass : ℝ}
    (hC : 0 ≤ C)
    (htotal : (∑ t : T, radius t) ≤ ∑ b ∈ active, bucketLoss b)
    (hbucket : ∀ b ∈ active,
      bucketLoss b ≤ C * Real.sqrt (bucketCount b))
    (hcount : ∀ b ∈ active, 0 ≤ bucketCount b)
    (hcount_sum : (∑ b ∈ active, bucketCount b) = horizonMass) :
    (∑ t : T, radius t)
      ≤ C * Real.sqrt ((active.card : ℝ) * horizonMass) := by
  have hloss :
      (∑ b ∈ active, bucketLoss b)
        ≤ ∑ b ∈ active, C * Real.sqrt (bucketCount b) := by
    exact Finset.sum_le_sum (fun b hb => hbucket b hb)
  have hfactor :
      (∑ b ∈ active, C * Real.sqrt (bucketCount b))
        = C * ∑ b ∈ active, Real.sqrt (bucketCount b) := by
    rw [Finset.mul_sum]
  have hsqrt :
      (∑ b ∈ active, Real.sqrt (bucketCount b))
        ≤ Real.sqrt ((active.card : ℝ) * horizonMass) := by
    simpa [hcount_sum] using
      sum_sqrt_le_sqrt_finset_card_mul_sum active bucketCount hcount
  have hmul :
      C * (∑ b ∈ active, Real.sqrt (bucketCount b))
        ≤ C * Real.sqrt ((active.card : ℝ) * horizonMass) :=
    mul_le_mul_of_nonneg_left hsqrt hC
  linarith

/-! ## Active-bucket concentration events -/

variable {I A : Type*} [Fintype I] [DecidableEq A]

/-- Active-bucket concentration event.  Only buckets in `active` contribute
to the statistical envelope; this is the non-vacuous replacement for a
full-action `|A|` union bound. -/
def ActiveBucketConcentrationEvent
    (active : Finset Bkt)
    (radius : T → ℝ) (bucketLoss bucketCount : Bkt → ℝ)
    (C horizonMass : ℝ) : Prop :=
  0 ≤ C ∧
    (∑ t : T, radius t) ≤ ∑ b ∈ active, bucketLoss b ∧
    (∀ b ∈ active, bucketLoss b ≤ C * Real.sqrt (bucketCount b)) ∧
    (∀ b ∈ active, 0 ≤ bucketCount b) ∧
    (∑ b ∈ active, bucketCount b) = horizonMass

/-- Active-bucket concentration gives a confidence-radius envelope depending
on `active.card`, not the full global action count. -/
theorem active_bucket_concentration_event_radius_envelope
    (active : Finset Bkt)
    (radius : T → ℝ) (bucketLoss bucketCount : Bkt → ℝ)
    {C horizonMass : ℝ}
    (hevent :
      ActiveBucketConcentrationEvent active radius bucketLoss bucketCount
        C horizonMass) :
    (∑ t : T, radius t)
      ≤ C * Real.sqrt ((active.card : ℝ) * horizonMass) := by
  rcases hevent with ⟨hC, htotal, hbucket, hcount, hsum⟩
  exact cumulative_radius_from_active_bucket_sqrt_bounds
    active radius bucketLoss bucketCount hC htotal hbucket hcount hsum

/-- Cumulative LCB learning regret with an active-bucket statistical envelope.
This is the theorem that replaces the vacuous
`sqrt(|A_full||Z|T)` scaling by `sqrt(|B_active|T)`. -/
theorem lcb_learning_regret_from_active_bucket_concentration_event
    (active : Finset Bkt)
    (F : ActionFamily A)
    (μtrue μhat : T → A → ServiceVec I)
    (rad : T → A → ℝ)
    (Q : T → ServiceVec I) (ε : T → ℝ)
    (bucketLoss bucketCount : Bkt → ℝ)
    {Qmax C horizonMass : ℝ}
    (hQ : ∀ t : T, Nonnegative (Q t))
    (hε : ∀ t : T, 0 ≤ ε t)
    (hpath : LCBConfidencePathEvent F μtrue μhat rad ε)
    (hbucket :
      ActiveBucketConcentrationEvent active ε bucketLoss bucketCount
        C horizonMass)
    (hQmax : ∀ t : T, l1 (Q t) ≤ Qmax)
    (hQmax_nonneg : 0 ≤ Qmax) :
    (∑ t : T,
      (support F (μtrue t) (Q t)
        - support F (lcbService (μhat t) (rad t)) (Q t)))
      ≤
    2 * Qmax * C * Real.sqrt ((active.card : ℝ) * horizonMass) := by
  rcases hpath with ⟨hconf, hrad⟩
  have henv :
      (∑ t : T, ε t)
        ≤ C * Real.sqrt ((active.card : ℝ) * horizonMass) :=
    active_bucket_concentration_event_radius_envelope
      active ε bucketLoss bucketCount hbucket
  have hbase := cumulative_lcb_learning_support_regret
    F μtrue μhat rad Q ε hQ hε hconf hrad
  have hterm :
      (∑ t : T, (2 * ε t) * l1 (Q t))
        ≤ ∑ t : T, (2 * ε t) * Qmax := by
    apply Finset.sum_le_sum
    intro t _
    exact mul_le_mul_of_nonneg_left (hQmax t) (by nlinarith [hε t])
  have hfactor :
      (∑ t : T, (2 * ε t) * Qmax)
        = 2 * Qmax * (∑ t : T, ε t) := by
    calc
      (∑ t : T, (2 * ε t) * Qmax)
          = ∑ t : T, (2 * Qmax) * ε t := by
            apply Finset.sum_congr rfl
            intro t _
            ring
      _ = (2 * Qmax) * (∑ t : T, ε t) := by
            rw [Finset.mul_sum]
      _ = 2 * Qmax * (∑ t : T, ε t) := by
            ring
  have hmul :
      2 * Qmax * (∑ t : T, ε t)
        ≤ 2 * Qmax
            * (C * Real.sqrt ((active.card : ℝ) * horizonMass)) := by
    exact mul_le_mul_of_nonneg_left henv (by nlinarith)
  have htarget :
      2 * Qmax
            * (C * Real.sqrt ((active.card : ℝ) * horizonMass))
        = 2 * Qmax * C
            * Real.sqrt ((active.card : ℝ) * horizonMass) := by
    ring
  linarith

/-! ## High-probability lifting

The concentration theorem itself belongs to the eventual adaptive sampling
model.  Once that model proves the active-bucket input event with probability
at least `p`, the deterministic regret theorem above lifts directly to a
high-probability learning-regret statement.
-/

/-- Event `E` has probability at least `p`.  The event is kept as a predicate
so later adaptive-sampling theorems can plug in their own measurable
construction without changing the regret layer. -/
def EventProbabilityAtLeast {Ω : Type*} [MeasurableSpace Ω]
    (ℙ : Measure Ω) (E : Ω → Prop) (p : ENNReal) : Prop :=
  p ≤ ℙ {ω | E ω}

/-- Event `E` fails with probability at most `δ`.  This failure-probability
form is convenient for finite active-bucket union bounds, because the theorem
does not need to commit to a particular `1 - δ` notation. -/
def EventFailureProbabilityAtMost {Ω : Type*} [MeasurableSpace Ω]
    (ℙ : Measure Ω) (E : Ω → Prop) (δ : ENNReal) : Prop :=
  ℙ {ω | ¬ E ω} ≤ δ

/-- Probability lower bounds are monotone under event implication. -/
theorem event_probability_mono {Ω : Type*} [MeasurableSpace Ω]
    (ℙ : Measure Ω) {E G : Ω → Prop} {p : ENNReal}
    (hE : EventProbabilityAtLeast ℙ E p)
    (himp : ∀ ω, E ω → G ω) :
    EventProbabilityAtLeast ℙ G p := by
  unfold EventProbabilityAtLeast at *
  exact le_trans hE (measure_mono (by
    intro ω hω
    exact himp ω hω))

/-- Finite active-bucket union bound.  If each active bucket's local event
fails with probability at most `δ b`, then the event that all active bucket
events hold fails with probability at most `Σ_{b∈active} δ b`. -/
theorem active_finset_all_events_failure_probability
    {Ω B : Type*} [MeasurableSpace Ω]
    (ℙ : Measure Ω) (active : Finset B)
    (E : B → Ω → Prop) (δ : B → ENNReal)
    (hfail : ∀ b ∈ active,
      EventFailureProbabilityAtMost ℙ (E b) (δ b)) :
    EventFailureProbabilityAtMost ℙ
      (fun ω => ∀ b ∈ active, E b ω)
      (∑ b ∈ active, δ b) := by
  unfold EventFailureProbabilityAtMost at *
  have hsubset : {ω | ¬ (∀ b ∈ active, E b ω)}
      ⊆ ⋃ b ∈ active, {ω | ¬ E b ω} := by
    intro ω hω
    push_neg at hω
    rcases hω with ⟨b, hb, hEb⟩
    exact Set.mem_iUnion₂.mpr ⟨b, hb, hEb⟩
  calc
    ℙ {ω | ¬ (∀ b ∈ active, E b ω)}
        ≤ ℙ (⋃ b ∈ active, {ω | ¬ E b ω}) :=
          measure_mono hsubset
    _ ≤ ∑ b ∈ active, ℙ {ω | ¬ E b ω} := by
          exact measure_biUnion_finset_le active (fun b => {ω | ¬ E b ω})
    _ ≤ ∑ b ∈ active, δ b := by
          exact Finset.sum_le_sum (fun b hb => hfail b hb)

/-- The complete deterministic input event needed by the active-bucket LCB
regret theorem.  A later probability model only has to prove this event with
high probability. -/
def ActiveBucketLCBInputEvent
    (active : Finset Bkt)
    (F : ActionFamily A)
    (μtrue μhat : T → A → ServiceVec I)
    (rad : T → A → ℝ)
    (Q : T → ServiceVec I) (ε : T → ℝ)
    (bucketLoss bucketCount : Bkt → ℝ)
    (Qmax C horizonMass : ℝ) : Prop :=
  (∀ t : T, Nonnegative (Q t)) ∧
    (∀ t : T, 0 ≤ ε t) ∧
    LCBConfidencePathEvent F μtrue μhat rad ε ∧
    ActiveBucketConcentrationEvent active ε bucketLoss bucketCount
      C horizonMass ∧
    (∀ t : T, l1 (Q t) ≤ Qmax) ∧
    0 ≤ Qmax

/-- The output event: cumulative LCB regret is bounded by the structured
active-bucket envelope. -/
def ActiveBucketLCBRegretBoundEvent
    (active : Finset Bkt)
    (F : ActionFamily A)
    (μtrue μhat : T → A → ServiceVec I)
    (rad : T → A → ℝ)
    (Q : T → ServiceVec I) (ε : T → ℝ)
    (Qmax C horizonMass : ℝ) : Prop :=
  (∑ t : T,
    (support F (μtrue t) (Q t)
      - support F (lcbService (μhat t) (rad t)) (Q t)))
    ≤
  2 * Qmax * C * Real.sqrt ((active.card : ℝ) * horizonMass)

/-- The active-bucket input event deterministically implies the active-bucket
LCB regret bound event. -/
theorem active_bucket_lcb_input_event_implies_regret_bound_event
    (active : Finset Bkt)
    (F : ActionFamily A)
    (μtrue μhat : T → A → ServiceVec I)
    (rad : T → A → ℝ)
    (Q : T → ServiceVec I) (ε : T → ℝ)
    (bucketLoss bucketCount : Bkt → ℝ)
    (Qmax C horizonMass : ℝ)
    (hevent :
      ActiveBucketLCBInputEvent active F μtrue μhat rad Q ε
        bucketLoss bucketCount Qmax C horizonMass) :
    ActiveBucketLCBRegretBoundEvent active F μtrue μhat rad Q ε
      Qmax C horizonMass := by
  rcases hevent with ⟨hQ, hε, hpath, hbucket, hQmax, hQmax_nonneg⟩
  exact lcb_learning_regret_from_active_bucket_concentration_event
    active F μtrue μhat rad Q ε bucketLoss bucketCount
    hQ hε hpath hbucket hQmax hQmax_nonneg

/-- High-probability active-bucket regret theorem.  The only remaining
probabilistic obligation is to prove that the concrete adaptive sampler
satisfies `ActiveBucketLCBInputEvent` with probability at least `p`. -/
theorem active_bucket_lcb_learning_regret_high_probability
    {Ω : Type*} [MeasurableSpace Ω]
    (ℙ : Measure Ω) {p : ENNReal}
    (active : Finset Bkt)
    (F : ActionFamily A)
    (μtrue μhat : Ω → T → A → ServiceVec I)
    (rad : Ω → T → A → ℝ)
    (Q : Ω → T → ServiceVec I) (ε : Ω → T → ℝ)
    (bucketLoss bucketCount : Ω → Bkt → ℝ)
    (Qmax C horizonMass : Ω → ℝ)
    (hprob :
      EventProbabilityAtLeast ℙ
        (fun ω =>
          ActiveBucketLCBInputEvent active F (μtrue ω) (μhat ω)
            (rad ω) (Q ω) (ε ω) (bucketLoss ω) (bucketCount ω)
            (Qmax ω) (C ω) (horizonMass ω)) p) :
    EventProbabilityAtLeast ℙ
      (fun ω =>
        ActiveBucketLCBRegretBoundEvent active F (μtrue ω) (μhat ω)
          (rad ω) (Q ω) (ε ω) (Qmax ω) (C ω) (horizonMass ω)) p := by
  exact event_probability_mono ℙ hprob (fun ω hω =>
    active_bucket_lcb_input_event_implies_regret_bound_event
      active F (μtrue ω) (μhat ω) (rad ω) (Q ω) (ε ω)
      (bucketLoss ω) (bucketCount ω) (Qmax ω) (C ω) (horizonMass ω)
      hω)

end Scheduleurm


/-! ## Source: Scheduleurm/HausdorffCapacity.lean -/


/-!
# Scheduleurm: capacity-region Hausdorff / support-distance bounds

This file formalises the candidate-set capacity-region distance theorem.

For compact convex capacity regions, uniform support-function distance is the
standard dual representation of Hausdorff distance.  The finite-action
configuration model already works with support functions because that is
exactly what MaxWeight optimises.  We therefore state the capacity-region
distance in this operational support-function form:

`|H_full(q) - H_cand(q)| ≤ ε` for every nonnegative `q` with `||q||₁ ≤ 1`.

The one-sided full-to-candidate inequality is induced by a topology/interference
cover and service Lipschitzness.  The reverse inequality is exact when the
candidate family is a subset of the full family.
-/

noncomputable section

set_option linter.unusedSectionVars false
set_option linter.unusedVariables false

namespace Scheduleurm

variable {I A : Type*} [Fintype I] [DecidableEq A]

/-- One-sided support-function Hausdorff bound for capacity regions in
nonnegative queue-pressure directions. -/
def CapacityDirectedSupportDistanceAtMost
    (full cand : ActionFamily A) (μ : A → ServiceVec I) (η : ℝ) : Prop :=
  ∀ q : ServiceVec I, Nonnegative q → l1 q ≤ 1 →
    support full μ q ≤ support cand μ q + η

/-- Symmetric support-function Hausdorff bound for the full and candidate
capacity regions, restricted to nonnegative queue-pressure directions. -/
def CapacitySupportHausdorffAtMost
    (full cand : ActionFamily A) (μ : A → ServiceVec I) (η : ℝ) : Prop :=
  CapacityDirectedSupportDistanceAtMost full cand μ η ∧
    CapacityDirectedSupportDistanceAtMost cand full μ η

/-- A support-gap bound `H_full ≤ H_cand + ε||q||₁` induces a normalized
capacity-region support-distance bound `≤ ε`. -/
theorem support_gap_implies_capacity_directed_support_distance
    (full cand : ActionFamily A) (μ : A → ServiceVec I)
    {ε : ℝ}
    (hε : 0 ≤ ε)
    (hgap : SupportGapAtMost full cand μ ε) :
    CapacityDirectedSupportDistanceAtMost full cand μ ε := by
  intro q hq hnorm
  have h := hgap q hq
  have hloss : ε * l1 q ≤ ε := by
    nlinarith [mul_le_mul_of_nonneg_left hnorm hε]
  linarith

/-- Candidate inclusion gives the reverse support inequality at zero loss. -/
theorem subset_implies_capacity_directed_support_distance_zero
    (cand full : ActionFamily A) (μ : A → ServiceVec I)
    (hsubset : ActionFamily.Subset cand full) :
    CapacityDirectedSupportDistanceAtMost cand full μ 0 := by
  intro q _hq _hnorm
  simpa using support_mono cand full μ q hsubset

/-- If candidates are a subset of the full action family and the full family
has support loss at most `ε` relative to candidates, then the two capacity
regions are within `ε` in normalized support/Hausdorff distance. -/
theorem support_gap_capacity_support_hausdorff
    (full cand : ActionFamily A) (μ : A → ServiceVec I)
    {ε : ℝ}
    (hε : 0 ≤ ε)
    (hsubset : ActionFamily.Subset cand full)
    (hgap : SupportGapAtMost full cand μ ε) :
    CapacitySupportHausdorffAtMost full cand μ ε := by
  constructor
  · exact support_gap_implies_capacity_directed_support_distance
      full cand μ hε hgap
  · intro q hq hnorm
    have hzero := subset_implies_capacity_directed_support_distance_zero
      cand full μ hsubset q hq hnorm
    linarith

/-- Metric/topology cover plus Lipschitz service gives the capacity-region
support/Hausdorff distance theorem.  This is the formal candidate-set
robustness result corresponding to
`d_H(Λ_full, Λ_cand) ≤ Lρ`. -/
theorem metric_cover_capacity_support_hausdorff
    (full cand : ActionFamily A) (μ : A → ServiceVec I)
    (d : A → A → ℝ) {L ρ : ℝ}
    (hL : 0 ≤ L) (hρ : 0 ≤ ρ)
    (hsubset : ActionFamily.Subset cand full)
    (hmetric : MetricCandidateCovers full cand d ρ)
    (hlip : ServiceLipschitz μ d L) :
    CapacitySupportHausdorffAtMost full cand μ (L * ρ) := by
  have hgap : SupportGapAtMost full cand μ (L * ρ) :=
    metric_cover_support_gap full cand μ d hL hρ hmetric hlip
  exact support_gap_capacity_support_hausdorff full cand μ
    (mul_nonneg hL hρ) hsubset hgap

/-- The same Hausdorff/support-distance bound preserves capacity slack up to
`ε` in every normalized queue-pressure direction. -/
theorem capacity_support_hausdorff_preserves_slack
    (full cand : ActionFamily A) (μ : A → ServiceVec I)
    {lam q : ServiceVec I} {δ ε : ℝ}
    (hq : Nonnegative q) (hnorm : l1 q ≤ 1)
    (hcap : InCapacityWithSlack full μ lam δ)
    (hdist : CapacityDirectedSupportDistanceAtMost full cand μ ε) :
    dot q lam + δ * l1 q ≤ support cand μ q + ε := by
  have hfull : dot q lam + δ * l1 q ≤ support full μ q :=
    capacity_slack_implies_support_slack full μ hq hcap
  have hdistq : support full μ q ≤ support cand μ q + ε :=
    hdist q hq hnorm
  exact le_trans hfull hdistq

end Scheduleurm


/-! ## Source: Scheduleurm/CapacityGeometry.lean -/


/-!
# Scheduleurm: capacity-set geometry and Hausdorff boundary

This file separates three notions that are often conflated in prose:

1. vertex/service-vector covering;
2. normalized support-function Hausdorff distance, the object used by
   MaxWeight;
3. metric Hausdorff distance between convex capacity sets.

For finite-dimensional compact convex sets, (2) and (3) are related by the
standard support-function/Hausdorff duality theorem.  Mathlib does not expose
that theorem in the exact form needed here, so we name it as a precise bridge
assumption instead of silently replacing it with an unrelated statement.
-/

noncomputable section

set_option linter.unusedSectionVars false
set_option linter.unusedVariables false

namespace Scheduleurm

open BigOperators

variable {I A : Type*} [Fintype I] [DecidableEq I] [DecidableEq A]

/-- Coordinatewise `ℓ∞` closeness for service vectors. -/
def CoordinateClose (v w : ServiceVec I) (η : ℝ) : Prop :=
  ∀ i : I, |v i - w i| ≤ η

/-- One-sided service-vector Hausdorff cover at the vertex/action level. -/
def ServiceVectorDirectedHausdorffAtMost
    (full cand : ActionFamily A) (μ : A → ServiceVec I) (η : ℝ) : Prop :=
  ∀ a ∈ full.acts, ∃ b ∈ cand.acts, CoordinateClose (μ a) (μ b) η

/-- Candidate service-vector cover is exactly a directed Hausdorff bound on
the service vertices. -/
theorem candidate_cover_service_vector_directed_hausdorff
    (full cand : ActionFamily A) (μ : A → ServiceVec I) {η : ℝ}
    (hcover : CandidateCovers full cand μ η) :
    ServiceVectorDirectedHausdorffAtMost full cand μ η := by
  intro a ha
  rcases hcover a ha with ⟨b, hb, hcoord⟩
  exact ⟨b, hb, hcoord⟩

/-- Metric/topology cover plus Lipschitz service gives a directed Hausdorff
bound on service vertices. -/
theorem metric_cover_service_vector_directed_hausdorff
    (full cand : ActionFamily A) (μ : A → ServiceVec I)
    (d : A → A → ℝ) {L ρ : ℝ}
    (hL : 0 ≤ L)
    (hmetric : MetricCandidateCovers full cand d ρ)
    (hlip : ServiceLipschitz μ d L) :
    ServiceVectorDirectedHausdorffAtMost full cand μ (L * ρ) := by
  have hcover : CandidateCovers full cand μ (L * ρ) :=
    candidate_cover_from_metric_lipschitz full cand μ d hL hmetric hlip
  exact candidate_cover_service_vector_directed_hausdorff full cand μ hcover

/-- Convex capacity set generated by stationary mixes over global
configuration actions. -/
def capacitySet (F : ActionFamily A) (μ : A → ServiceVec I) :
    Set (ServiceVec I) :=
  {v | ∃ x : StationaryMix F, v = mixedService F μ x}

/-! ## Constructive capacity-set closeness

The support-function result is the MaxWeight-facing theorem.  For reviewers
who ask for a metric capacity-set statement, the following results prove a
direct coordinatewise Hausdorff bound by pushing each stationary mixture
through an explicit action cover.  This avoids using compact-convex
support/Hausdorff duality as a black-box assumption.
-/

/-- Push a stationary randomized mix on `full` through an action map whose
image lies in `cand`. -/
def pushforwardMix (full cand : ActionFamily A)
    (x : StationaryMix full) (π : A → A)
    (hπ : ∀ a ∈ full.acts, π a ∈ cand.acts) : StationaryMix cand where
  weight b := ∑ a ∈ full.acts, if π a = b then x.weight a else 0
  nonneg_on := by
    intro b hb
    apply Finset.sum_nonneg
    intro a ha
    by_cases h : π a = b
    · simp [h, x.nonneg_on a ha]
    · simp [h]
  sum_one := by
    calc
      (∑ b ∈ cand.acts, ∑ a ∈ full.acts,
          if π a = b then x.weight a else 0)
          = ∑ a ∈ full.acts, ∑ b ∈ cand.acts,
              if π a = b then x.weight a else 0 := by
            rw [Finset.sum_comm]
      _ = ∑ a ∈ full.acts, x.weight a := by
            apply Finset.sum_congr rfl
            intro a ha
            have hmem : π a ∈ cand.acts := hπ a ha
            have hsum :
                (∑ b ∈ cand.acts, if π a = b then x.weight a else 0)
                  = x.weight a := by
              rw [Finset.sum_ite_eq cand.acts (π a) (fun _ => x.weight a)]
              simp [hmem]
            exact hsum
      _ = 1 := x.sum_one

/-- Mixed service after pushing a mix forward is the pushforward weighted
service average. -/
lemma mixedService_pushforward_eq
    (full cand : ActionFamily A) (μ : A → ServiceVec I)
    (x : StationaryMix full) (π : A → A)
    (hπ : ∀ a ∈ full.acts, π a ∈ cand.acts) :
    mixedService cand μ (pushforwardMix full cand x π hπ)
      = fun i => ∑ a ∈ full.acts, x.weight a * μ (π a) i := by
  funext i
  unfold mixedService pushforwardMix
  calc
    (∑ b ∈ cand.acts,
        (∑ a ∈ full.acts, if π a = b then x.weight a else 0) * μ b i)
        = ∑ b ∈ cand.acts, ∑ a ∈ full.acts,
            (if π a = b then x.weight a else 0) * μ b i := by
          apply Finset.sum_congr rfl
          intro b hb
          rw [Finset.sum_mul]
    _ = ∑ a ∈ full.acts, ∑ b ∈ cand.acts,
            (if π a = b then x.weight a else 0) * μ b i := by
          rw [Finset.sum_comm]
    _ = ∑ a ∈ full.acts, x.weight a * μ (π a) i := by
          apply Finset.sum_congr rfl
          intro a ha
          have hmem : π a ∈ cand.acts := hπ a ha
          have hsum :
              (∑ b ∈ cand.acts,
                  (if π a = b then x.weight a else 0) * μ b i)
                = x.weight a * μ (π a) i := by
            calc
              (∑ b ∈ cand.acts,
                  (if π a = b then x.weight a else 0) * μ b i)
                  = ∑ b ∈ cand.acts,
                      if π a = b then x.weight a * μ b i else 0 := by
                    apply Finset.sum_congr rfl
                    intro b hb
                    by_cases h : π a = b <;> simp [h]
              _ = x.weight a * μ (π a) i := by
                    rw [Finset.sum_ite_eq cand.acts (π a)
                      (fun b => x.weight a * μ b i)]
                    simp [hmem]
          exact hsum

/-- If every action is mapped to a coordinate-close candidate action, then
the pushed-forward mixed service is coordinate-close by the same radius. -/
lemma mixedService_pushforward_coordinateClose
    (full cand : ActionFamily A) (μ : A → ServiceVec I)
    (x : StationaryMix full) (π : A → A)
    (hπ : ∀ a ∈ full.acts, π a ∈ cand.acts) {η : ℝ}
    (hη : 0 ≤ η)
    (hclose : ∀ a ∈ full.acts, CoordinateClose (μ a) (μ (π a)) η) :
    CoordinateClose (mixedService full μ x)
      (mixedService cand μ (pushforwardMix full cand x π hπ)) η := by
  intro i
  rw [mixedService_pushforward_eq full cand μ x π hπ]
  unfold mixedService
  have hdiff :
      (∑ a ∈ full.acts, x.weight a * μ a i)
        - (∑ a ∈ full.acts, x.weight a * μ (π a) i)
        = ∑ a ∈ full.acts, x.weight a * (μ a i - μ (π a) i) := by
    rw [← Finset.sum_sub_distrib]
    apply Finset.sum_congr rfl
    intro a ha
    ring
  rw [hdiff]
  calc
    |∑ a ∈ full.acts, x.weight a * (μ a i - μ (π a) i)|
        ≤ ∑ a ∈ full.acts, |x.weight a * (μ a i - μ (π a) i)| :=
          Finset.abs_sum_le_sum_abs _ _
    _ = ∑ a ∈ full.acts, x.weight a * |μ a i - μ (π a) i| := by
          apply Finset.sum_congr rfl
          intro a ha
          rw [abs_mul, abs_of_nonneg (x.nonneg_on a ha)]
    _ ≤ ∑ a ∈ full.acts, x.weight a * η := by
          apply Finset.sum_le_sum
          intro a ha
          exact mul_le_mul_of_nonneg_left
            (hclose a ha i) (x.nonneg_on a ha)
    _ = (∑ a ∈ full.acts, x.weight a) * η := by
          rw [Finset.sum_mul]
    _ = η := by
          rw [x.sum_one, one_mul]

/-- Directed coordinate-Hausdorff capacity-set bound: every full-region
service vector has a candidate-region representative within `η` in every
queue coordinate. -/
def CapacitySetDirectedCoordinateHausdorffAtMost
    (full cand : ActionFamily A) (μ : A → ServiceVec I) (η : ℝ) : Prop :=
  ∀ v : ServiceVec I, v ∈ capacitySet full μ →
    ∃ w : ServiceVec I, w ∈ capacitySet cand μ ∧ CoordinateClose v w η

/-- Symmetric coordinate-Hausdorff capacity-set bound. -/
def CapacitySetCoordinateHausdorffAtMost
    (full cand : ActionFamily A) (μ : A → ServiceVec I) (η : ℝ) : Prop :=
  CapacitySetDirectedCoordinateHausdorffAtMost full cand μ η ∧
    CapacitySetDirectedCoordinateHausdorffAtMost cand full μ η

/-- A concrete action map induces a directed coordinate-Hausdorff bound
between the corresponding capacity sets. -/
theorem capacitySet_directed_coordinate_hausdorff_of_action_map
    (full cand : ActionFamily A) (μ : A → ServiceVec I)
    (π : A → A) {η : ℝ}
    (hη : 0 ≤ η)
    (hπ : ∀ a ∈ full.acts, π a ∈ cand.acts)
    (hclose : ∀ a ∈ full.acts, CoordinateClose (μ a) (μ (π a)) η) :
    CapacitySetDirectedCoordinateHausdorffAtMost full cand μ η := by
  intro v hv
  rcases hv with ⟨x, rfl⟩
  let y := pushforwardMix full cand x π hπ
  refine ⟨mixedService cand μ y, ?_, ?_⟩
  · exact ⟨y, rfl⟩
  · exact mixedService_pushforward_coordinateClose full cand μ x π hπ hη hclose

/-- A service-vector vertex cover lifts constructively to a capacity-set
coordinate-Hausdorff bound by pushing stationary mixtures through the selected
nearest candidate vertices. -/
theorem capacitySet_directed_coordinate_hausdorff_of_vertex_cover
    (full cand : ActionFamily A) (μ : A → ServiceVec I) {η : ℝ}
    (hη : 0 ≤ η)
    (hcover : ServiceVectorDirectedHausdorffAtMost full cand μ η) :
    CapacitySetDirectedCoordinateHausdorffAtMost full cand μ η := by
  classical
  let π : A → A :=
    fun a => if ha : a ∈ full.acts
      then Classical.choose (hcover a ha)
      else cand.nonempty.choose
  have hπ : ∀ a ∈ full.acts, π a ∈ cand.acts := by
    intro a ha
    simp [π, ha, (Classical.choose_spec (hcover a ha)).1]
  have hclose : ∀ a ∈ full.acts, CoordinateClose (μ a) (μ (π a)) η := by
    intro a ha
    simp [π, ha]
    exact (Classical.choose_spec (hcover a ha)).2
  exact capacitySet_directed_coordinate_hausdorff_of_action_map
    full cand μ π hη hπ hclose

/-- If `cand` is a literal subfamily of `full`, the reverse directed
capacity-set error is zero. -/
theorem subset_implies_capacitySet_directed_coordinate_hausdorff_zero
    (cand full : ActionFamily A) (μ : A → ServiceVec I)
    (hsubset : ActionFamily.Subset cand full) :
    CapacitySetDirectedCoordinateHausdorffAtMost cand full μ 0 := by
  refine capacitySet_directed_coordinate_hausdorff_of_action_map
    cand full μ (fun a => a) (by norm_num) ?_ ?_
  · intro a ha
    exact hsubset ha
  · intro a ha i
    simp

/-- Service-vector cover plus candidate subset yields a symmetric
coordinate-Hausdorff bound between capacity sets.  The reverse direction is
exact because candidate policies are feasible full policies. -/
theorem service_vector_cover_capacitySet_coordinate_hausdorff
    (full cand : ActionFamily A) (μ : A → ServiceVec I) {η : ℝ}
    (hη : 0 ≤ η)
    (hsubset : ActionFamily.Subset cand full)
    (hcover : ServiceVectorDirectedHausdorffAtMost full cand μ η) :
    CapacitySetCoordinateHausdorffAtMost full cand μ η := by
  constructor
  · exact capacitySet_directed_coordinate_hausdorff_of_vertex_cover
      full cand μ hη hcover
  · intro v hv
    rcases subset_implies_capacitySet_directed_coordinate_hausdorff_zero
      cand full μ hsubset v hv with ⟨w, hw, hclose0⟩
    refine ⟨w, hw, ?_⟩
    intro i
    have h0 := hclose0 i
    linarith

/-- Fabric metric cover plus service Lipschitzness yields a constructive
coordinate-Hausdorff bound between the candidate and full capacity sets,
without invoking compact-convex support/Hausdorff duality. -/
theorem metric_cover_capacitySet_coordinate_hausdorff
    (full cand : ActionFamily A) (μ : A → ServiceVec I)
    (d : A → A → ℝ) {L ρ : ℝ}
    (hL : 0 ≤ L) (hρ : 0 ≤ ρ)
    (hsubset : ActionFamily.Subset cand full)
    (hmetric : MetricCandidateCovers full cand d ρ)
    (hlip : ServiceLipschitz μ d L) :
    CapacitySetCoordinateHausdorffAtMost full cand μ (L * ρ) := by
  have hvertex :
      ServiceVectorDirectedHausdorffAtMost full cand μ (L * ρ) :=
    metric_cover_service_vector_directed_hausdorff
      full cand μ d hL hmetric hlip
  exact service_vector_cover_capacitySet_coordinate_hausdorff
    full cand μ (mul_nonneg hL hρ) hsubset hvertex

/-- Metric Hausdorff distance bound between convex capacity sets in the
ambient finite-dimensional service-vector metric. -/
def CapacityMetricHausdorffAtMost
    (full cand : ActionFamily A) (μ : A → ServiceVec I) (η : ℝ) : Prop :=
  Metric.hausdorffDist (capacitySet full μ) (capacitySet cand μ) ≤ η

/-- The exact convex-geometry bridge needed to rewrite support-distance
results as metric Hausdorff results.  This is a named assumption because the
full theorem requires finite-dimensional compact convex support-function
duality, which is outside the local scheduler algebra. -/
def CapacitySupportMetricHausdorffDuality
    (full cand : ActionFamily A) (μ : A → ServiceVec I) (η : ℝ) : Prop :=
  CapacitySupportHausdorffAtMost full cand μ η →
    CapacityMetricHausdorffAtMost full cand μ η

/-- Under the standard compact-convex support/Hausdorff duality theorem, the
metric/topology candidate cover yields the metric Hausdorff capacity-region
bound `d_H(Λ_full, Λ_cand) ≤ Lρ`. -/
theorem metric_cover_capacity_metric_hausdorff
    (full cand : ActionFamily A) (μ : A → ServiceVec I)
    (d : A → A → ℝ) {L ρ : ℝ}
    (hL : 0 ≤ L) (hρ : 0 ≤ ρ)
    (hsubset : ActionFamily.Subset cand full)
    (hmetric : MetricCandidateCovers full cand d ρ)
    (hlip : ServiceLipschitz μ d L)
    (hduality :
      CapacitySupportMetricHausdorffDuality full cand μ (L * ρ)) :
    CapacityMetricHausdorffAtMost full cand μ (L * ρ) := by
  have hsupport : CapacitySupportHausdorffAtMost full cand μ (L * ρ) :=
    metric_cover_capacity_support_hausdorff full cand μ d
      hL hρ hsubset hmetric hlip
  exact hduality hsupport

end Scheduleurm


/-! ## Source: Scheduleurm/OperationalMetric.lean -/


/-!
# Scheduleurm: operational topology/interference metrics

The candidate-cover theorem should not rely on a mysterious abstract metric.
This file provides a concrete finite-feature metric template.  A feature can
encode fabric placement, co-location profile, network path class, memory
pressure, or other scheduler-visible descriptors; the distance is a weighted
`ℓ₁` distance over those descriptors.
-/

noncomputable section

set_option linter.unusedSectionVars false
set_option linter.unusedVariables false

namespace Scheduleurm

open BigOperators

variable {I A D : Type*} [Fintype I] [DecidableEq A] [Fintype D]

/-- Finite feature representation of global configuration actions. -/
structure FabricFeature (A D : Type*) where
  feature : A → D → ℝ
  weight : D → ℝ
  weight_nonneg : ∀ d : D, 0 ≤ weight d

/-- Weighted `ℓ₁` fabric/interference distance between configurations. -/
def fabricDistance (Φ : FabricFeature A D) (a b : A) : ℝ :=
  ∑ d : D, Φ.weight d * |Φ.feature a d - Φ.feature b d|

/-- Fabric distance is nonnegative. -/
lemma fabricDistance_nonneg (Φ : FabricFeature A D) (a b : A) :
    0 ≤ fabricDistance Φ a b := by
  unfold fabricDistance
  exact Finset.sum_nonneg
    (fun d _ => mul_nonneg (Φ.weight_nonneg d) (abs_nonneg _))

/-- Service is Lipschitz with respect to the operational fabric metric. -/
def FabricServiceLipschitz
    (μ : A → ServiceVec I) (Φ : FabricFeature A D) (L : ℝ) : Prop :=
  ServiceLipschitz μ (fabricDistance Φ) L

/-- Candidate actions cover full actions under the operational fabric metric. -/
def FabricCandidateCovers
    (full cand : ActionFamily A) (Φ : FabricFeature A D) (ρ : ℝ) : Prop :=
  MetricCandidateCovers full cand (fabricDistance Φ) ρ

/-! ## Calibration certificates

The paper should not treat `d_Φ`, `L`, and `ρ` as magic assumptions.  The
following certificate layer connects finite telemetry features and a concrete
candidate projection to the abstract cover/Lipschitz hypotheses used by the
capacity and drift theorems.
-/

/-- A concrete projection from each full action to a candidate action within
fabric radius `ρ`.  This is the checkable form of a candidate generator:
given a full configuration, the generator supplies a nearby candidate. -/
structure FabricCandidateProjection
    (full cand : ActionFamily A) (Φ : FabricFeature A D) (ρ : ℝ) where
  project : A → A
  project_mem : ∀ a ∈ full.acts, project a ∈ cand.acts
  dist_le : ∀ a ∈ full.acts, fabricDistance Φ a (project a) ≤ ρ

/-- A concrete projection certificate implies the fabric-cover hypothesis. -/
theorem FabricCandidateProjection.covers
    (full cand : ActionFamily A) (Φ : FabricFeature A D) (ρ : ℝ)
    (π : FabricCandidateProjection full cand Φ ρ) :
    FabricCandidateCovers full cand Φ ρ := by
  intro a ha
  exact ⟨π.project a, π.project_mem a ha, π.dist_le a ha⟩

/-- Feature-level service sensitivity envelope.  The coefficient `c i d`
measures how much service coordinate `i` can change per unit change in
fabric feature `d`.  This is the formal target for profiling and perturbation
experiments. -/
def FeatureServiceSensitivityEnvelope
    (μ : A → ServiceVec I) (Φ : FabricFeature A D)
    (c : I → D → ℝ) : Prop :=
  ∀ a b i,
    |μ a i - μ b i|
      ≤ ∑ d : D, c i d * |Φ.feature a d - Φ.feature b d|

/-- The profiled feature coefficients are dominated by the weighted fabric
metric with global constant `L`. -/
def FeatureSensitivityDominatedByFabricMetric
    (Φ : FabricFeature A D) (c : I → D → ℝ) (L : ℝ) : Prop :=
  ∀ i d, c i d ≤ L * Φ.weight d

/-- Feature-level sensitivity plus coefficient domination proves the global
fabric-service Lipschitz condition used by the candidate approximation
theorem. -/
theorem fabric_service_lipschitz_of_feature_sensitivity
    (μ : A → ServiceVec I) (Φ : FabricFeature A D)
    (c : I → D → ℝ) (L : ℝ)
    (henv : FeatureServiceSensitivityEnvelope μ Φ c)
    (hdom : FeatureSensitivityDominatedByFabricMetric Φ c L) :
    FabricServiceLipschitz μ Φ L := by
  intro a b i
  calc
    |μ a i - μ b i|
        ≤ ∑ d : D, c i d * |Φ.feature a d - Φ.feature b d| :=
          henv a b i
    _ ≤ ∑ d : D, (L * Φ.weight d) * |Φ.feature a d - Φ.feature b d| := by
          apply Finset.sum_le_sum
          intro d _
          exact mul_le_mul_of_nonneg_right (hdom i d) (abs_nonneg _)
    _ = L * fabricDistance Φ a b := by
          unfold fabricDistance
          rw [Finset.mul_sum]
          apply Finset.sum_congr rfl
          intro d _
          ring

/-- A fully calibrated fabric certificate gives the support-function
candidate approximation used in the main stability theorem. -/
theorem calibrated_fabric_cover_support_gap
    (full cand : ActionFamily A) (μ : A → ServiceVec I)
    (Φ : FabricFeature A D) (c : I → D → ℝ) {L ρ : ℝ}
    (hL : 0 ≤ L) (hρ : 0 ≤ ρ)
    (π : FabricCandidateProjection full cand Φ ρ)
    (henv : FeatureServiceSensitivityEnvelope μ Φ c)
    (hdom : FeatureSensitivityDominatedByFabricMetric Φ c L) :
    SupportGapAtMost full cand μ (L * ρ) := by
  exact metric_cover_support_gap full cand μ (fabricDistance Φ) hL hρ
    (π.covers full cand Φ ρ)
    (fabric_service_lipschitz_of_feature_sensitivity μ Φ c L henv hdom)

/-- State/regime-indexed calibrated support gap.  This is the precise form
needed when feasible action families depend on queue state, available jobs,
or hidden regime: each index supplies its own full/candidate families,
projection certificate, and profiled feature sensitivity. -/
theorem indexed_calibrated_fabric_cover_support_gap
    {S : Type*}
    (full cand : S → ActionFamily A)
    (μ : S → A → ServiceVec I)
    (Φ : S → FabricFeature A D)
    (c : S → I → D → ℝ)
    (L ρ : S → ℝ)
    (hL : ∀ s : S, 0 ≤ L s)
    (hρ : ∀ s : S, 0 ≤ ρ s)
    (π : ∀ s : S, FabricCandidateProjection (full s) (cand s) (Φ s) (ρ s))
    (henv : ∀ s : S, FeatureServiceSensitivityEnvelope (μ s) (Φ s) (c s))
    (hdom : ∀ s : S, FeatureSensitivityDominatedByFabricMetric (Φ s) (c s) (L s)) :
    ∀ s : S, SupportGapAtMost (full s) (cand s) (μ s) (L s * ρ s) := by
  intro s
  exact calibrated_fabric_cover_support_gap
    (full s) (cand s) (μ s) (Φ s) (c s)
    (hL s) (hρ s) (π s) (henv s) (hdom s)

/-- Uniform constant version of the state/regime-indexed calibrated support
gap.  It is useful when the paper assumes one global `L` and `ρ` across all
regimes or feasible-family states. -/
theorem indexed_calibrated_fabric_cover_support_gap_uniform
    {S : Type*}
    (full cand : S → ActionFamily A)
    (μ : S → A → ServiceVec I)
    (Φ : S → FabricFeature A D)
    (c : S → I → D → ℝ)
    {L ρ : ℝ}
    (hL : 0 ≤ L) (hρ : 0 ≤ ρ)
    (π : ∀ s : S, FabricCandidateProjection (full s) (cand s) (Φ s) ρ)
    (henv : ∀ s : S, FeatureServiceSensitivityEnvelope (μ s) (Φ s) (c s))
    (hdom : ∀ s : S, FeatureSensitivityDominatedByFabricMetric (Φ s) (c s) L) :
    ∀ s : S, SupportGapAtMost (full s) (cand s) (μ s) (L * ρ) := by
  exact indexed_calibrated_fabric_cover_support_gap
    full cand μ Φ c (fun _ => L) (fun _ => ρ)
    (fun _ => hL) (fun _ => hρ) π henv hdom

/-- The finite-feature operational metric instantiates the generic candidate
support-gap theorem. -/
theorem fabric_cover_support_gap
    (full cand : ActionFamily A) (μ : A → ServiceVec I)
    (Φ : FabricFeature A D) {L ρ : ℝ}
    (hL : 0 ≤ L) (hρ : 0 ≤ ρ)
    (hcover : FabricCandidateCovers full cand Φ ρ)
    (hlip : FabricServiceLipschitz μ Φ L) :
    SupportGapAtMost full cand μ (L * ρ) := by
  exact metric_cover_support_gap full cand μ (fabricDistance Φ)
    hL hρ hcover hlip

/-- Operational fabric cover gives the normalized capacity support-Hausdorff
candidate approximation theorem. -/
theorem fabric_cover_capacity_support_hausdorff
    (full cand : ActionFamily A) (μ : A → ServiceVec I)
    (Φ : FabricFeature A D) {L ρ : ℝ}
    (hL : 0 ≤ L) (hρ : 0 ≤ ρ)
    (hsubset : ActionFamily.Subset cand full)
    (hcover : FabricCandidateCovers full cand Φ ρ)
    (hlip : FabricServiceLipschitz μ Φ L) :
    CapacitySupportHausdorffAtMost full cand μ (L * ρ) := by
  exact metric_cover_capacity_support_hausdorff
    full cand μ (fabricDistance Φ) hL hρ hsubset hcover hlip

end Scheduleurm


/-! ## Source: Scheduleurm/MainTheorems.lean -/


/-!
# Scheduleurm: narrowed paper-level theorem spine

This file responds to the reviewer-style recommendation to narrow the paper
around three core mathematical claims:

1. candidate-restricted capacity approximation;
2. robust candidate MaxWeight drift/stability with explicit slack accounting;
3. operational stochastic stability under a verified queue model.

Hidden regimes and learning are kept as explicit extension layers rather than
being silently folded into the main theorem.
-/

noncomputable section

set_option linter.unusedSectionVars false
set_option linter.unusedVariables false

namespace Scheduleurm

variable {I A D Ω T K Bkt : Type*}
  [Fintype I] [DecidableEq I] [DecidableEq A] [Fintype D] [Fintype Ω]
  [Fintype T] [Fintype K] [DecidableEq K] [DecidableEq Bkt]

/-- **Main theorem 1: candidate-restricted capacity approximation.**

A finite-feature operational fabric cover plus service Lipschitzness yields
the normalized capacity support-Hausdorff loss `Lρ`. -/
theorem main_candidate_restricted_capacity_approximation
    (full cand : ActionFamily A) (μ : A → ServiceVec I)
    (Φ : FabricFeature A D) {L ρ : ℝ}
    (hL : 0 ≤ L) (hρ : 0 ≤ ρ)
    (hsubset : ActionFamily.Subset cand full)
    (hcover : FabricCandidateCovers full cand Φ ρ)
    (hlip : FabricServiceLipschitz μ Φ L) :
    CapacitySupportHausdorffAtMost full cand μ (L * ρ) := by
  exact fabric_cover_capacity_support_hausdorff
    full cand μ Φ hL hρ hsubset hcover hlip

/-- Constructive coordinate-Hausdorff version of Main theorem 1.  This is a
metric capacity-set statement proved by pushing stationary mixtures through
the fabric cover, so it does not rely on a separate compact-convex
support/Hausdorff duality assumption. -/
theorem main_candidate_restricted_capacity_coordinate_hausdorff
    (full cand : ActionFamily A) (μ : A → ServiceVec I)
    (Φ : FabricFeature A D) {L ρ : ℝ}
    (hL : 0 ≤ L) (hρ : 0 ≤ ρ)
    (hsubset : ActionFamily.Subset cand full)
    (hcover : FabricCandidateCovers full cand Φ ρ)
    (hlip : FabricServiceLipschitz μ Φ L) :
    CapacitySetCoordinateHausdorffAtMost full cand μ (L * ρ) := by
  exact metric_cover_capacitySet_coordinate_hausdorff
    full cand μ (fabricDistance Φ) hL hρ hsubset hcover hlip

/-- Candidate capacity approximation from calibrated fabric certificates.
Instead of assuming `L`-Lipschitz service and a `ρ`-cover abstractly, this
version takes a concrete candidate projection and feature-sensitivity
envelope.  It is the theorem-level bridge from profiling certificates to the
support-function capacity loss `Lρ`. -/
theorem main_candidate_restricted_capacity_approximation_from_calibration
    (full cand : ActionFamily A) (μ : A → ServiceVec I)
    (Φ : FabricFeature A D) (c : I → D → ℝ) {L ρ : ℝ}
    (hL : 0 ≤ L) (hρ : 0 ≤ ρ)
    (hsubset : ActionFamily.Subset cand full)
    (π : FabricCandidateProjection full cand Φ ρ)
    (henv : FeatureServiceSensitivityEnvelope μ Φ c)
    (hdom : FeatureSensitivityDominatedByFabricMetric Φ c L) :
    CapacitySupportHausdorffAtMost full cand μ (L * ρ) := by
  exact fabric_cover_capacity_support_hausdorff full cand μ Φ
    hL hρ hsubset (π.covers full cand Φ ρ)
    (fabric_service_lipschitz_of_feature_sensitivity μ Φ c L henv hdom)

/-- Constructive coordinate-Hausdorff capacity approximation from calibrated
fabric certificates. -/
theorem main_candidate_restricted_capacity_coordinate_hausdorff_from_calibration
    (full cand : ActionFamily A) (μ : A → ServiceVec I)
    (Φ : FabricFeature A D) (c : I → D → ℝ) {L ρ : ℝ}
    (hL : 0 ≤ L) (hρ : 0 ≤ ρ)
    (hsubset : ActionFamily.Subset cand full)
    (π : FabricCandidateProjection full cand Φ ρ)
    (henv : FeatureServiceSensitivityEnvelope μ Φ c)
    (hdom : FeatureSensitivityDominatedByFabricMetric Φ c L) :
    CapacitySetCoordinateHausdorffAtMost full cand μ (L * ρ) := by
  exact main_candidate_restricted_capacity_coordinate_hausdorff
    full cand μ Φ hL hρ hsubset (π.covers full cand Φ ρ)
    (fabric_service_lipschitz_of_feature_sensitivity μ Φ c L henv hdom)

/-- Paper-facing support consequence for the downward-closed capacity region.
This is the queueing capacity object used in the text: load can be strictly
below a stationary average service vector because unused service may be
wasted. -/
theorem main_downward_capacity_support_slack
    (F : ActionFamily A) (μ : A → ServiceVec I)
    {lam q : ServiceVec I} {δ : ℝ}
    (hq : Nonnegative q)
    (hcap : InDownwardCapacityWithSlack F μ lam δ) :
    dot q lam + δ * l1 q ≤ support F μ q := by
  exact downward_capacity_slack_implies_support_slack F μ hq hcap

/-- **Main theorem 2: robust candidate MaxWeight drift with explicit slack.**

Candidate approximation, estimation error, and queue-growing penalty rate all
consume capacity slack.  The remaining margin is
`δ - (εcand + εest + β)`. -/
theorem main_robust_candidate_maxweight_drift
    (full cand : ActionFamily A)
    (μtrue lower : A → ServiceVec I)
    {Q lam : ServiceVec I} {δ εcand εest B P0 β : ℝ}
    (hQ : Nonnegative Q) (hlam : Nonnegative lam)
    (hcap : InCapacityWithSlack full μtrue lam δ)
    (hgap : SupportGapAtMost full cand μtrue εcand)
    (hlower_gap : support cand μtrue Q
        ≤ support cand lower Q + εest * l1 Q)
    (penalty : ServiceVec I → A → ℝ) {astar : A}
    (hmax : RobustScoreMaximizer cand lower Q (penalty Q) astar)
    (hpen : QueueScaledPenaltyBounded cand penalty P0 β)
    (hSecond : secondOrderTerm lam (lower astar) ≤ B)
    (hlower_a_nonneg : Nonnegative (lower astar)) :
    Lyapunov (queueStep Q lam (lower astar)) - Lyapunov Q
      ≤ B + P0 - (δ - (εcand + εest + β)) * l1 Q := by
  exact robust_candidate_policy_lyapunov_drift_scaled_penalty
    full cand μtrue lower hQ hlam hcap hgap hlower_gap penalty
    hmax hpen hSecond hlower_a_nonneg

/-- Robust candidate MaxWeight drift with a queue-scaled approximate
optimization oracle.  The oracle's backlog-proportional error `α1` consumes
capacity slack; its bounded error `α0` increases the additive drift constant.
-/
theorem main_robust_candidate_maxweight_drift_approx_oracle
    (full cand : ActionFamily A)
    (μtrue lower : A → ServiceVec I)
    {Q lam : ServiceVec I} {δ εcand εest B P0 β α0 α1 : ℝ}
    (hQ : Nonnegative Q) (hlam : Nonnegative lam)
    (hcap : InCapacityWithSlack full μtrue lam δ)
    (hgap : SupportGapAtMost full cand μtrue εcand)
    (hlower_gap : support cand μtrue Q
        ≤ support cand lower Q + εest * l1 Q)
    (penalty : ServiceVec I → A → ℝ) {astar : A}
    (hmax :
      QueueScaledApproxRobustScoreMaximizer cand lower Q penalty astar α0 α1)
    (hpen : QueueScaledPenaltyBounded cand penalty P0 β)
    (hSecond : secondOrderTerm lam (lower astar) ≤ B)
    (hlower_a_nonneg : Nonnegative (lower astar)) :
    Lyapunov (queueStep Q lam (lower astar)) - Lyapunov Q
      ≤ B + (P0 + α0)
          - (δ - (εcand + εest + β + α1)) * l1 Q := by
  exact robust_candidate_policy_lyapunov_drift_scaled_penalty_approx_oracle
    full cand μtrue lower hQ hlam hcap hgap hlower_gap penalty
    hmax hpen hSecond hlower_a_nonneg

/-- Diagonal-scaled lower-service support theorem.

This paper-facing theorem is the stochastic LCB bridge for heterogeneous
Scheduleurm workloads.  Instead of charging one absolute error budget across
CPU, GPU, CNN, LLM, and hybrid classes, each class has its own service scale.
The theorem remains a support-function statement, so it plugs into the same
robust MaxWeight drift wrapper through `hlower_gap`. -/
theorem main_diagonal_scaled_lcb_support_loss
    (F : ActionFamily A) (μtrue μhat rad : A → ServiceVec I)
    (scale : ServiceVec I)
    {q : ServiceVec I} {ε : ℝ}
    (hq : Nonnegative q) (hε : 0 ≤ ε) (hscale : Nonnegative scale)
    (hconf : CoordinateServiceConfidenceEvent F μtrue μhat rad)
    (hrad : CoordinateRadiusBoundedByScale F rad scale ε) :
    support F μtrue q
      ≤ support F (coordinateLcbService μhat rad) q
          + (2 * ε) * weightedPressure scale q := by
  exact true_support_le_coordinateLcb_support_plus_scaled_estimation_error
    F μtrue μhat rad scale hq hε hscale hconf hrad

/-- Drift-norm version of diagonal-scaled LCB support loss.  If the selected
normalizers are uniformly bounded by `C`, the lower-service support loss can
be consumed by the standard margin as `εest = 2εC`. -/
theorem main_diagonal_scaled_lcb_support_loss_l1
    (F : ActionFamily A) (μtrue μhat rad : A → ServiceVec I)
    (scale : ServiceVec I)
    {q : ServiceVec I} {ε C : ℝ}
    (hq : Nonnegative q) (hε : 0 ≤ ε) (hscale : Nonnegative scale)
    (hscale_bound : ∀ i : I, scale i ≤ C)
    (hconf : CoordinateServiceConfidenceEvent F μtrue μhat rad)
    (hrad : CoordinateRadiusBoundedByScale F rad scale ε) :
    support F μtrue q
      ≤ support F (coordinateLcbService μhat rad) q
          + ((2 * ε) * C) * l1 q := by
  exact true_support_le_coordinateLcb_support_plus_scaled_estimation_error_l1
    F μtrue μhat rad scale hq hε hscale hscale_bound hconf hrad

/-- **Main theorem 3: operational stochastic stability under verified drift.**

For the integer queue model, a concrete stochastic system must verify
conditional drift domination and local return on a finite sublevel set.  Under
those model obligations, the queue is positive recurrent via that finite set. -/
theorem main_operational_stochastic_stability
    (M : NatQueueTransitionModel I)
    {B η α : ℝ} {N : ℕ}
    (hη : 0 < η) (hα : 0 < α)
    (hN : B + α ≤ η * (N : ℝ))
    (hdet : ∀ Q : NatQueueState I,
      M.deterministicDrift Q ≤ B - η * natQueueL1 Q)
    (hreturn : ∀ Q : NatQueueState I,
      natQueueSmallSet (I := I) N Q →
        FiniteExpectedReturnTimeToSet M.K (natQueueSmallSet (I := I) N) Q) :
    PositiveRecurrentViaFiniteSet M.K (natQueueSmallSet (I := I) N) := by
  exact nat_model_linear_drift_positive_recurrent_via_finite_set
    M hη hα hN hdet hreturn

/-- Operational stochastic stability with finite-small-set local return
derived from the same Foster drift certificate. -/
theorem main_operational_stochastic_stability'
    (M : NatQueueTransitionModel I)
    {B η α : ℝ} {N : ℕ}
    (hη : 0 < η) (hα : 0 < α)
    (hN : B + α ≤ η * (N : ℝ))
    (hdet : ∀ Q : NatQueueState I,
      M.deterministicDrift Q ≤ B - η * natQueueL1 Q) :
    PositiveRecurrentViaFiniteSet M.K (natQueueSmallSet (I := I) N) := by
  exact nat_model_linear_drift_positive_recurrent_via_finite_set'
    M hη hα hN hdet

/-- Operational necessity from a concrete conservation law: every
operationally stabilizable load has a stationary occupation measure whose
average service dominates arrivals, hence lies in the zero-slack capacity
closure. -/
theorem main_operational_conservation_law_necessity
    (F : ActionFamily A) (μ : A → ServiceVec I)
    (hcons : CapacityRegionOperationalConservationLaw F μ) :
    CapacityRegionOperationalNecessary F μ := by
  exact conservation_law_implies_operational_necessity F μ hcons

/-- Paper-level operational capacity sandwich: interior capacity implies
operational stabilizability under verified drift, while operational
stabilizability implies zero-slack capacity-closure membership under the
conservation law. -/
theorem main_operational_capacity_sandwich
    (F : ActionFamily A) (μ : A → ServiceVec I)
    (hsound : CapacityRegionOperationalSound F μ)
    (hcons : CapacityRegionOperationalConservationLaw F μ) :
    (∀ lam δ,
      InCapacityWithSlack F μ lam δ →
        OperationallyStabilizesIntegerLoad (I := I) lam) ∧
      (∀ lam,
        OperationallyStabilizesIntegerLoad (I := I) lam →
          InCapacityWithSlack F μ lam 0) := by
  exact configuration_capacity_region_operational_sandwich
    F μ hsound (conservation_law_implies_operational_necessity F μ hcons)

/-- Concrete version of Main theorem 3 for the finite-support arrival/service
model.  Here `drift_dominated` is not assumed as a field supplied by the
paper author; it is proved from the explicit finite-sample transition model,
second-order bound, arrival-pressure bound, and service-pressure lower bound.
-/
theorem main_concrete_finite_support_stochastic_stability
    (M : FiniteSupportQueueModel I Ω)
    (B : NatQueueState I → ℝ)
    (lambda lower : NatQueueState I → ServiceVec I)
    (hsecond : ∀ Q : NatQueueState I,
      M.expectedSecondOrder Q ≤ B Q)
    (harrival : ∀ Q : NatQueueState I,
      (∑ ω : Ω, M.prob Q ω *
          dot (natQueueToServiceVec Q)
            (natQueueToServiceVec (M.arrivals Q ω)))
        ≤ dot (natQueueToServiceVec Q) (lambda Q))
    (hservice : ∀ Q : NatQueueState I,
      dot (natQueueToServiceVec Q) (lower Q)
        ≤ ∑ ω : Ω, M.prob Q ω *
            dot (natQueueToServiceVec Q)
              (natQueueToServiceVec (M.service Q ω)))
    {B0 η α : ℝ} {N : ℕ}
    (hη : 0 < η) (hα : 0 < α)
    (hN : B0 + α ≤ η * (N : ℝ))
    (hlinear : ∀ Q : NatQueueState I,
      B Q + dot (natQueueToServiceVec Q) (lambda Q)
          - dot (natQueueToServiceVec Q) (lower Q)
        ≤ B0 - η * natQueueL1 Q)
    (hreturn : ∀ Q : NatQueueState I,
      natQueueSmallSet (I := I) N Q →
        FiniteExpectedReturnTimeToSet M.K (natQueueSmallSet (I := I) N) Q) :
    PositiveRecurrentViaFiniteSet M.K (natQueueSmallSet (I := I) N) := by
  exact M.positive_recurrent_via_finite_set
    B lambda lower hsecond harrival hservice hη hα hN hlinear hreturn

/-- Concrete version of Main theorem 3 from coordinate-level stochastic
moments.  The arrival and service pressure bounds are proved, not assumed,
from conditional coordinate means. -/
theorem main_concrete_finite_support_stochastic_stability_from_coordinate_moments
    (M : FiniteSupportQueueModel I Ω)
    (B : NatQueueState I → ℝ)
    (lambda lower : NatQueueState I → ServiceVec I)
    (hsecond : ∀ Q : NatQueueState I, M.expectedSecondOrder Q ≤ B Q)
    (harrival_coord : ∀ Q : NatQueueState I, ∀ i : I,
      M.expectedArrival Q i ≤ lambda Q i)
    (hservice_coord : ∀ Q : NatQueueState I, ∀ i : I,
      lower Q i ≤ M.expectedService Q i)
    {B0 η α : ℝ} {N : ℕ}
    (hη : 0 < η) (hα : 0 < α)
    (hN : B0 + α ≤ η * (N : ℝ))
    (hlinear : ∀ Q : NatQueueState I,
      B Q + dot (natQueueToServiceVec Q) (lambda Q)
          - dot (natQueueToServiceVec Q) (lower Q)
        ≤ B0 - η * natQueueL1 Q)
    (hreturn : ∀ Q : NatQueueState I,
      natQueueSmallSet (I := I) N Q →
        FiniteExpectedReturnTimeToSet M.K (natQueueSmallSet (I := I) N) Q) :
    PositiveRecurrentViaFiniteSet M.K (natQueueSmallSet (I := I) N) := by
  exact M.positive_recurrent_via_coordinate_moments
    B lambda lower hsecond harrival_coord hservice_coord hη hα hN
    hlinear hreturn

/-- Concrete robust-candidate stochastic stability theorem.

This is the fully connected version of the proof spine: the model has an
explicit finite-support arrival/service law, coordinate conditional moments
bound arrivals and selected service, the full action space has capacity
slack, and the selected candidate action is a robust-score maximizer.  The
stochastic `drift_dominated` obligation is discharged inside the theorem; the
remaining slack is exactly `δ - (εcand + εest + β)`.
-/
theorem main_concrete_robust_candidate_stochastic_stability
    (M : FiniteSupportQueueModel I Ω)
    (full cand : ActionFamily A)
    (μtrue lower : A → ServiceVec I)
    (lam : ServiceVec I)
    (penalty : ServiceVec I → A → ℝ)
    (astar : NatQueueState I → A)
    {δ εcand εest B P0 β α : ℝ} {N : ℕ}
    (hcap : InCapacityWithSlack full μtrue lam δ)
    (hgap : SupportGapAtMost full cand μtrue εcand)
    (hlower_gap : ∀ q : ServiceVec I, Nonnegative q →
      support cand μtrue q ≤ support cand lower q + εest * l1 q)
    (hmax : ∀ Q : NatQueueState I,
      RobustScoreMaximizer cand lower (natQueueToServiceVec Q)
        (penalty (natQueueToServiceVec Q)) (astar Q))
    (hpen : QueueScaledPenaltyBounded cand penalty P0 β)
    (hsecond : ∀ Q : NatQueueState I, M.expectedSecondOrder Q ≤ B)
    (harrival_coord : ∀ Q : NatQueueState I, ∀ i : I,
      M.expectedArrival Q i ≤ lam i)
    (hservice_coord : ∀ Q : NatQueueState I, ∀ i : I,
      lower (astar Q) i ≤ M.expectedService Q i)
    (hmargin : 0 < δ - (εcand + εest + β))
    (hα : 0 < α)
    (hN : B + P0 + α ≤ (δ - (εcand + εest + β)) * (N : ℝ))
    (hreturn : ∀ Q : NatQueueState I,
      natQueueSmallSet (I := I) N Q →
        FiniteExpectedReturnTimeToSet M.K (natQueueSmallSet (I := I) N) Q) :
    PositiveRecurrentViaFiniteSet M.K (natQueueSmallSet (I := I) N) := by
  let selectedLower : NatQueueState I → ServiceVec I :=
    fun Q => lower (astar Q)
  have hlinear : ∀ Q : NatQueueState I,
      B + dot (natQueueToServiceVec Q) lam
          - dot (natQueueToServiceVec Q) (selectedLower Q)
        ≤ B + P0 - (δ - (εcand + εest + β)) * natQueueL1 Q := by
    intro Q
    let q : ServiceVec I := natQueueToServiceVec Q
    have hq : Nonnegative q := natQueueToServiceVec_nonnegative Q
    have hslack : dot q lam + δ * l1 q ≤ support full μtrue q :=
      capacity_slack_implies_support_slack full μtrue hq hcap
    have happrox : support full μtrue q
        ≤ dot q (lower (astar Q))
            + (εcand + εest + β) * l1 q + P0 :=
      robust_candidate_policy_approx_full_support_scaled_penalty
        full cand μtrue lower hq hgap (hlower_gap q hq) penalty
        (hmax Q) hpen
    have hl1 : l1 q = natQueueL1 Q := l1_natQueueToServiceVec Q
    unfold selectedLower
    change B + dot q lam - dot q (lower (astar Q))
      ≤ B + P0 - (δ - (εcand + εest + β)) * natQueueL1 Q
    rw [← hl1]
    linarith
  exact M.positive_recurrent_via_coordinate_moments
    (fun _ : NatQueueState I => B)
    (fun _ : NatQueueState I => lam)
    selectedLower
    hsecond
    (by
      intro Q i
      exact harrival_coord Q i)
    (by
      intro Q i
      exact hservice_coord Q i)
    hmargin hα hN hlinear hreturn

/-- Fabric-cover version of the concrete robust stochastic stability theorem.

The candidate gap is not assumed abstractly here.  It is proved from a
finite-feature fabric/interference metric: the maintained candidate action
family is a `ρ`-cover of the full global configuration action family, and the
true service map is `L`-Lipschitz under that metric.  The slack consumed by
restricting to candidates is therefore exactly `L * ρ`.
-/
theorem main_concrete_fabric_cover_robust_candidate_stochastic_stability
    (M : FiniteSupportQueueModel I Ω)
    (full cand : ActionFamily A)
    (μtrue lower : A → ServiceVec I)
    (Φ : FabricFeature A D)
    (lam : ServiceVec I)
    (penalty : ServiceVec I → A → ℝ)
    (astar : NatQueueState I → A)
    {L ρ δ εest B P0 β α : ℝ} {N : ℕ}
    (hL : 0 ≤ L) (hρ : 0 ≤ ρ)
    (hcover : FabricCandidateCovers full cand Φ ρ)
    (hlip : FabricServiceLipschitz μtrue Φ L)
    (hcap : InCapacityWithSlack full μtrue lam δ)
    (hlower_gap : ∀ q : ServiceVec I, Nonnegative q →
      support cand μtrue q ≤ support cand lower q + εest * l1 q)
    (hmax : ∀ Q : NatQueueState I,
      RobustScoreMaximizer cand lower (natQueueToServiceVec Q)
        (penalty (natQueueToServiceVec Q)) (astar Q))
    (hpen : QueueScaledPenaltyBounded cand penalty P0 β)
    (hsecond : ∀ Q : NatQueueState I, M.expectedSecondOrder Q ≤ B)
    (harrival_coord : ∀ Q : NatQueueState I, ∀ i : I,
      M.expectedArrival Q i ≤ lam i)
    (hservice_coord : ∀ Q : NatQueueState I, ∀ i : I,
      lower (astar Q) i ≤ M.expectedService Q i)
    (hmargin : 0 < δ - (L * ρ + εest + β))
    (hα : 0 < α)
    (hN : B + P0 + α ≤ (δ - (L * ρ + εest + β)) * (N : ℝ))
    (hreturn : ∀ Q : NatQueueState I,
      natQueueSmallSet (I := I) N Q →
        FiniteExpectedReturnTimeToSet M.K (natQueueSmallSet (I := I) N) Q) :
    PositiveRecurrentViaFiniteSet M.K (natQueueSmallSet (I := I) N) := by
  have hgap : SupportGapAtMost full cand μtrue (L * ρ) :=
    fabric_cover_support_gap full cand μtrue Φ hL hρ hcover hlip
  exact main_concrete_robust_candidate_stochastic_stability
    M full cand μtrue lower lam penalty astar hcap hgap hlower_gap
    hmax hpen hsecond harrival_coord hservice_coord hmargin hα hN hreturn

/-- Bounded-sample version of the fabric-cover robust stochastic theorem.

Here the quadratic drift constant is proved from coordinatewise bounds on
every one-step arrival and service realization.  Thus the stochastic side of
the theorem is stated in primitive scheduler/profiling terms:
bounded batches, coordinate conditional mean arrivals, coordinate conditional
mean selected service, and local return on the finite sublevel set.
-/
theorem main_concrete_fabric_cover_robust_candidate_stochastic_stability_from_bounded_samples
    (M : FiniteSupportQueueModel I Ω)
    (full cand : ActionFamily A)
    (μtrue lower : A → ServiceVec I)
    (Φ : FabricFeature A D)
    (lam : ServiceVec I)
    (penalty : ServiceVec I → A → ℝ)
    (astar : NatQueueState I → A)
    (Amax Smax : ServiceVec I)
    {L ρ δ εest P0 β α : ℝ} {N : ℕ}
    (hAmax : Nonnegative Amax) (hSmax : Nonnegative Smax)
    (hArr_bound : ∀ Q : NatQueueState I, ∀ ω : Ω, ∀ i : I,
      (M.arrivals Q ω i : ℝ) ≤ Amax i)
    (hServ_bound : ∀ Q : NatQueueState I, ∀ ω : Ω, ∀ i : I,
      (M.service Q ω i : ℝ) ≤ Smax i)
    (hL : 0 ≤ L) (hρ : 0 ≤ ρ)
    (hcover : FabricCandidateCovers full cand Φ ρ)
    (hlip : FabricServiceLipschitz μtrue Φ L)
    (hcap : InCapacityWithSlack full μtrue lam δ)
    (hlower_gap : ∀ q : ServiceVec I, Nonnegative q →
      support cand μtrue q ≤ support cand lower q + εest * l1 q)
    (hmax : ∀ Q : NatQueueState I,
      RobustScoreMaximizer cand lower (natQueueToServiceVec Q)
        (penalty (natQueueToServiceVec Q)) (astar Q))
    (hpen : QueueScaledPenaltyBounded cand penalty P0 β)
    (harrival_coord : ∀ Q : NatQueueState I, ∀ i : I,
      M.expectedArrival Q i ≤ lam i)
    (hservice_coord : ∀ Q : NatQueueState I, ∀ i : I,
      lower (astar Q) i ≤ M.expectedService Q i)
    (hmargin : 0 < δ - (L * ρ + εest + β))
    (hα : 0 < α)
    (hN : secondOrderTerm Amax Smax + P0 + α
      ≤ (δ - (L * ρ + εest + β)) * (N : ℝ))
    (hreturn : ∀ Q : NatQueueState I,
      natQueueSmallSet (I := I) N Q →
        FiniteExpectedReturnTimeToSet M.K (natQueueSmallSet (I := I) N) Q) :
    PositiveRecurrentViaFiniteSet M.K (natQueueSmallSet (I := I) N) := by
  exact main_concrete_fabric_cover_robust_candidate_stochastic_stability
    M full cand μtrue lower Φ lam penalty astar hL hρ hcover hlip
    hcap hlower_gap hmax hpen
    (M.expectedSecondOrder_le_of_coord_sample_bounds
      Amax Smax hAmax hSmax hArr_bound hServ_bound)
    harrival_coord hservice_coord hmargin hα hN hreturn

/-- Concrete robust-candidate stochastic stability with local return derived
from the Foster drift certificate. -/
theorem main_concrete_robust_candidate_stochastic_stability'
    (M : FiniteSupportQueueModel I Ω)
    (full cand : ActionFamily A)
    (μtrue lower : A → ServiceVec I)
    (lam : ServiceVec I)
    (penalty : ServiceVec I → A → ℝ)
    (astar : NatQueueState I → A)
    {δ εcand εest B P0 β α : ℝ} {N : ℕ}
    (hcap : InCapacityWithSlack full μtrue lam δ)
    (hgap : SupportGapAtMost full cand μtrue εcand)
    (hlower_gap : ∀ q : ServiceVec I, Nonnegative q →
      support cand μtrue q ≤ support cand lower q + εest * l1 q)
    (hmax : ∀ Q : NatQueueState I,
      RobustScoreMaximizer cand lower (natQueueToServiceVec Q)
        (penalty (natQueueToServiceVec Q)) (astar Q))
    (hpen : QueueScaledPenaltyBounded cand penalty P0 β)
    (hsecond : ∀ Q : NatQueueState I, M.expectedSecondOrder Q ≤ B)
    (harrival_coord : ∀ Q : NatQueueState I, ∀ i : I,
      M.expectedArrival Q i ≤ lam i)
    (hservice_coord : ∀ Q : NatQueueState I, ∀ i : I,
      lower (astar Q) i ≤ M.expectedService Q i)
    (hmargin : 0 < δ - (εcand + εest + β))
    (hα : 0 < α)
    (hN : B + P0 + α ≤ (δ - (εcand + εest + β)) * (N : ℝ)) :
    PositiveRecurrentViaFiniteSet M.K (natQueueSmallSet (I := I) N) := by
  let selectedLower : NatQueueState I → ServiceVec I :=
    fun Q => lower (astar Q)
  have hlinear : ∀ Q : NatQueueState I,
      B + dot (natQueueToServiceVec Q) lam
          - dot (natQueueToServiceVec Q) (selectedLower Q)
        ≤ B + P0 - (δ - (εcand + εest + β)) * natQueueL1 Q := by
    intro Q
    let q : ServiceVec I := natQueueToServiceVec Q
    have hq : Nonnegative q := natQueueToServiceVec_nonnegative Q
    have hslack : dot q lam + δ * l1 q ≤ support full μtrue q :=
      capacity_slack_implies_support_slack full μtrue hq hcap
    have happrox : support full μtrue q
        ≤ dot q (lower (astar Q))
            + (εcand + εest + β) * l1 q + P0 :=
      robust_candidate_policy_approx_full_support_scaled_penalty
        full cand μtrue lower hq hgap (hlower_gap q hq) penalty
        (hmax Q) hpen
    have hl1 : l1 q = natQueueL1 Q := l1_natQueueToServiceVec Q
    unfold selectedLower
    change B + dot q lam - dot q (lower (astar Q))
      ≤ B + P0 - (δ - (εcand + εest + β)) * natQueueL1 Q
    rw [← hl1]
    linarith
  exact M.positive_recurrent_via_coordinate_moments'
    (fun _ : NatQueueState I => B)
    (fun _ : NatQueueState I => lam)
    selectedLower
    hsecond
    (by
      intro Q i
      exact harrival_coord Q i)
    (by
      intro Q i
      exact hservice_coord Q i)
    hmargin hα hN hlinear

/-- Fabric-cover concrete robust stochastic stability with local return
derived from Foster drift. -/
theorem main_concrete_fabric_cover_robust_candidate_stochastic_stability'
    (M : FiniteSupportQueueModel I Ω)
    (full cand : ActionFamily A)
    (μtrue lower : A → ServiceVec I)
    (Φ : FabricFeature A D)
    (lam : ServiceVec I)
    (penalty : ServiceVec I → A → ℝ)
    (astar : NatQueueState I → A)
    {L ρ δ εest B P0 β α : ℝ} {N : ℕ}
    (hL : 0 ≤ L) (hρ : 0 ≤ ρ)
    (hcover : FabricCandidateCovers full cand Φ ρ)
    (hlip : FabricServiceLipschitz μtrue Φ L)
    (hcap : InCapacityWithSlack full μtrue lam δ)
    (hlower_gap : ∀ q : ServiceVec I, Nonnegative q →
      support cand μtrue q ≤ support cand lower q + εest * l1 q)
    (hmax : ∀ Q : NatQueueState I,
      RobustScoreMaximizer cand lower (natQueueToServiceVec Q)
        (penalty (natQueueToServiceVec Q)) (astar Q))
    (hpen : QueueScaledPenaltyBounded cand penalty P0 β)
    (hsecond : ∀ Q : NatQueueState I, M.expectedSecondOrder Q ≤ B)
    (harrival_coord : ∀ Q : NatQueueState I, ∀ i : I,
      M.expectedArrival Q i ≤ lam i)
    (hservice_coord : ∀ Q : NatQueueState I, ∀ i : I,
      lower (astar Q) i ≤ M.expectedService Q i)
    (hmargin : 0 < δ - (L * ρ + εest + β))
    (hα : 0 < α)
    (hN : B + P0 + α ≤ (δ - (L * ρ + εest + β)) * (N : ℝ)) :
    PositiveRecurrentViaFiniteSet M.K (natQueueSmallSet (I := I) N) := by
  have hgap : SupportGapAtMost full cand μtrue (L * ρ) :=
    fabric_cover_support_gap full cand μtrue Φ hL hρ hcover hlip
  exact main_concrete_robust_candidate_stochastic_stability'
    M full cand μtrue lower lam penalty astar hcap hgap hlower_gap
    hmax hpen hsecond harrival_coord hservice_coord hmargin hα hN

/-- Paper main theorem with a bounded conditional second-order moment rather
than coordinatewise bounded samples.  The finite-support Lean model still
supplies the expectation operator, but the theorem statement uses the
queueing-literature condition `expectedSecondOrder ≤ B`. -/
theorem main_theorem_robust_candidate_maxweight_stability_with_second_moment_bound
    (M : FiniteSupportQueueModel I Ω)
    (full cand : ActionFamily A)
    (μtrue lower : A → ServiceVec I)
    (Φ : FabricFeature A D)
    (lam : ServiceVec I)
    (penalty : ServiceVec I → A → ℝ)
    (astar : NatQueueState I → A)
    {L ρ δ εest B P0 β α : ℝ} {N : ℕ}
    (hL : 0 ≤ L) (hρ : 0 ≤ ρ)
    (hcover : FabricCandidateCovers full cand Φ ρ)
    (hlip : FabricServiceLipschitz μtrue Φ L)
    (hcap : InCapacityWithSlack full μtrue lam δ)
    (hlower_gap : ∀ q : ServiceVec I, Nonnegative q →
      support cand μtrue q ≤ support cand lower q + εest * l1 q)
    (hmax : ∀ Q : NatQueueState I,
      RobustScoreMaximizer cand lower (natQueueToServiceVec Q)
        (penalty (natQueueToServiceVec Q)) (astar Q))
    (hpen : QueueScaledPenaltyBounded cand penalty P0 β)
    (hsecond : ∀ Q : NatQueueState I, M.expectedSecondOrder Q ≤ B)
    (harrival_coord : ∀ Q : NatQueueState I, ∀ i : I,
      M.expectedArrival Q i ≤ lam i)
    (hservice_coord : ∀ Q : NatQueueState I, ∀ i : I,
      lower (astar Q) i ≤ M.expectedService Q i)
    (hmargin : 0 < δ - (L * ρ + εest + β))
    (hα : 0 < α)
    (hN : B + P0 + α
      ≤ (δ - (L * ρ + εest + β)) * (N : ℝ)) :
    PositiveRecurrentViaFiniteSet M.K (natQueueSmallSet (I := I) N) := by
  exact main_concrete_fabric_cover_robust_candidate_stochastic_stability'
    M full cand μtrue lower Φ lam penalty astar hL hρ hcover hlip
    hcap hlower_gap hmax hpen hsecond harrival_coord hservice_coord
    hmargin hα hN

/-- Paper main theorem with bounded conditional second-order moment and a
queue-scaled approximate optimization oracle.  This is the realistic solver
version: greedy, local-search, or time-limited ILP errors are charged as
`α0 + α1 ||Q||₁`. -/
theorem main_theorem_robust_candidate_maxweight_stability_with_second_moment_bound_approx_oracle
    (M : FiniteSupportQueueModel I Ω)
    (full cand : ActionFamily A)
    (μtrue lower : A → ServiceVec I)
    (Φ : FabricFeature A D)
    (lam : ServiceVec I)
    (penalty : ServiceVec I → A → ℝ)
    (astar : NatQueueState I → A)
    {L ρ δ εest B P0 β α0 α1 α : ℝ} {N : ℕ}
    (hL : 0 ≤ L) (hρ : 0 ≤ ρ)
    (hcover : FabricCandidateCovers full cand Φ ρ)
    (hlip : FabricServiceLipschitz μtrue Φ L)
    (hcap : InCapacityWithSlack full μtrue lam δ)
    (hlower_gap : ∀ q : ServiceVec I, Nonnegative q →
      support cand μtrue q ≤ support cand lower q + εest * l1 q)
    (hmax : ∀ Q : NatQueueState I,
      QueueScaledApproxRobustScoreMaximizer cand lower
        (natQueueToServiceVec Q) penalty (astar Q) α0 α1)
    (hpen : QueueScaledPenaltyBounded cand penalty P0 β)
    (hsecond : ∀ Q : NatQueueState I, M.expectedSecondOrder Q ≤ B)
    (harrival_coord : ∀ Q : NatQueueState I, ∀ i : I,
      M.expectedArrival Q i ≤ lam i)
    (hservice_coord : ∀ Q : NatQueueState I, ∀ i : I,
      lower (astar Q) i ≤ M.expectedService Q i)
    (hmargin : 0 < δ - (L * ρ + εest + β + α1))
    (hα : 0 < α)
    (hN : B + (P0 + α0) + α
      ≤ (δ - (L * ρ + εest + β + α1)) * (N : ℝ)) :
    PositiveRecurrentViaFiniteSet M.K (natQueueSmallSet (I := I) N) := by
  have hgap : SupportGapAtMost full cand μtrue (L * ρ) :=
    fabric_cover_support_gap full cand μtrue Φ hL hρ hcover hlip
  let selectedLower : NatQueueState I → ServiceVec I :=
    fun Q => lower (astar Q)
  have hlinear : ∀ Q : NatQueueState I,
      B + dot (natQueueToServiceVec Q) lam
          - dot (natQueueToServiceVec Q) (selectedLower Q)
        ≤ B + (P0 + α0)
            - (δ - (L * ρ + εest + β + α1)) * natQueueL1 Q := by
    intro Q
    let q : ServiceVec I := natQueueToServiceVec Q
    have hq : Nonnegative q := natQueueToServiceVec_nonnegative Q
    have hslack : dot q lam + δ * l1 q ≤ support full μtrue q :=
      capacity_slack_implies_support_slack full μtrue hq hcap
    have happrox : support full μtrue q
        ≤ dot q (lower (astar Q))
            + (L * ρ + εest + β + α1) * l1 q + (P0 + α0) :=
      robust_candidate_policy_approx_full_support_scaled_penalty_approx_oracle
        full cand μtrue lower hq hgap (hlower_gap q hq) penalty
        (hmax Q) hpen
    have hl1 : l1 q = natQueueL1 Q := l1_natQueueToServiceVec Q
    unfold selectedLower
    change B + dot q lam - dot q (lower (astar Q))
      ≤ B + (P0 + α0)
          - (δ - (L * ρ + εest + β + α1)) * natQueueL1 Q
    rw [← hl1]
    linarith
  exact M.positive_recurrent_via_coordinate_moments'
    (fun _ : NatQueueState I => B)
    (fun _ : NatQueueState I => lam)
    selectedLower
    hsecond
    (by
      intro Q i
      exact harrival_coord Q i)
    (by
      intro Q i
      exact hservice_coord Q i)
    hmargin hα hN hlinear

/-- Bounded-sample, fabric-cover, robust-candidate stochastic stability with
all stochastic drift and local-return obligations discharged by proof. -/
theorem main_concrete_fabric_cover_robust_candidate_stochastic_stability_from_bounded_samples'
    (M : FiniteSupportQueueModel I Ω)
    (full cand : ActionFamily A)
    (μtrue lower : A → ServiceVec I)
    (Φ : FabricFeature A D)
    (lam : ServiceVec I)
    (penalty : ServiceVec I → A → ℝ)
    (astar : NatQueueState I → A)
    (Amax Smax : ServiceVec I)
    {L ρ δ εest P0 β α : ℝ} {N : ℕ}
    (hAmax : Nonnegative Amax) (hSmax : Nonnegative Smax)
    (hArr_bound : ∀ Q : NatQueueState I, ∀ ω : Ω, ∀ i : I,
      (M.arrivals Q ω i : ℝ) ≤ Amax i)
    (hServ_bound : ∀ Q : NatQueueState I, ∀ ω : Ω, ∀ i : I,
      (M.service Q ω i : ℝ) ≤ Smax i)
    (hL : 0 ≤ L) (hρ : 0 ≤ ρ)
    (hcover : FabricCandidateCovers full cand Φ ρ)
    (hlip : FabricServiceLipschitz μtrue Φ L)
    (hcap : InCapacityWithSlack full μtrue lam δ)
    (hlower_gap : ∀ q : ServiceVec I, Nonnegative q →
      support cand μtrue q ≤ support cand lower q + εest * l1 q)
    (hmax : ∀ Q : NatQueueState I,
      RobustScoreMaximizer cand lower (natQueueToServiceVec Q)
        (penalty (natQueueToServiceVec Q)) (astar Q))
    (hpen : QueueScaledPenaltyBounded cand penalty P0 β)
    (harrival_coord : ∀ Q : NatQueueState I, ∀ i : I,
      M.expectedArrival Q i ≤ lam i)
    (hservice_coord : ∀ Q : NatQueueState I, ∀ i : I,
      lower (astar Q) i ≤ M.expectedService Q i)
    (hmargin : 0 < δ - (L * ρ + εest + β))
    (hα : 0 < α)
    (hN : secondOrderTerm Amax Smax + P0 + α
      ≤ (δ - (L * ρ + εest + β)) * (N : ℝ)) :
    PositiveRecurrentViaFiniteSet M.K (natQueueSmallSet (I := I) N) := by
  exact main_concrete_fabric_cover_robust_candidate_stochastic_stability'
    M full cand μtrue lower Φ lam penalty astar hL hρ hcover hlip
    hcap hlower_gap hmax hpen
    (M.expectedSecondOrder_le_of_coord_sample_bounds
      Amax Smax hAmax hSmax hArr_bound hServ_bound)
    harrival_coord hservice_coord hmargin hα hN

/-- **Paper main theorem: robust candidate MaxWeight stability under a
fabric-cover approximation.**

This is the single theorem a paper statement should cite.  It combines the
full-action support slack, candidate fabric-cover loss `L * ρ`, lower-service
estimation loss `εest`, queue-scaled switching/risk penalty `β`, bounded
finite-support arrival/service samples, coordinate conditional mean
dominance, and the Foster finite-set recurrence conclusion.

The mathematical content is:

`δ > L * ρ + εest + β`

implies positive recurrence through a finite backlog sublevel set, with the
quadratic drift constant derived from the sample bounds `Amax` and `Smax`.
-/
theorem main_theorem_robust_candidate_maxweight_stability_under_fabric_cover
    (M : FiniteSupportQueueModel I Ω)
    (full cand : ActionFamily A)
    (μtrue lower : A → ServiceVec I)
    (Φ : FabricFeature A D)
    (lam : ServiceVec I)
    (penalty : ServiceVec I → A → ℝ)
    (astar : NatQueueState I → A)
    (Amax Smax : ServiceVec I)
    {L ρ δ εest P0 β α : ℝ} {N : ℕ}
    (hAmax : Nonnegative Amax) (hSmax : Nonnegative Smax)
    (hArr_bound : ∀ Q : NatQueueState I, ∀ ω : Ω, ∀ i : I,
      (M.arrivals Q ω i : ℝ) ≤ Amax i)
    (hServ_bound : ∀ Q : NatQueueState I, ∀ ω : Ω, ∀ i : I,
      (M.service Q ω i : ℝ) ≤ Smax i)
    (hL : 0 ≤ L) (hρ : 0 ≤ ρ)
    (hcover : FabricCandidateCovers full cand Φ ρ)
    (hlip : FabricServiceLipschitz μtrue Φ L)
    (hcap : InCapacityWithSlack full μtrue lam δ)
    (hlower_gap : ∀ q : ServiceVec I, Nonnegative q →
      support cand μtrue q ≤ support cand lower q + εest * l1 q)
    (hmax : ∀ Q : NatQueueState I,
      RobustScoreMaximizer cand lower (natQueueToServiceVec Q)
        (penalty (natQueueToServiceVec Q)) (astar Q))
    (hpen : QueueScaledPenaltyBounded cand penalty P0 β)
    (harrival_coord : ∀ Q : NatQueueState I, ∀ i : I,
      M.expectedArrival Q i ≤ lam i)
    (hservice_coord : ∀ Q : NatQueueState I, ∀ i : I,
      lower (astar Q) i ≤ M.expectedService Q i)
    (hmargin : 0 < δ - (L * ρ + εest + β))
    (hα : 0 < α)
    (hN : secondOrderTerm Amax Smax + P0 + α
      ≤ (δ - (L * ρ + εest + β)) * (N : ℝ)) :
    PositiveRecurrentViaFiniteSet M.K (natQueueSmallSet (I := I) N) := by
  exact
    main_concrete_fabric_cover_robust_candidate_stochastic_stability_from_bounded_samples'
      M full cand μtrue lower Φ lam penalty astar Amax Smax
      hAmax hSmax hArr_bound hServ_bound hL hρ hcover hlip hcap
      hlower_gap hmax hpen harrival_coord hservice_coord hmargin hα hN

/-- Calibrated version of the paper main theorem.  The fabric-cover and
service-Lipschitz assumptions are derived from a concrete projection
certificate and feature-sensitivity envelope, so the only remaining numerical
obligations are the profiled constants themselves. -/
theorem main_theorem_robust_candidate_maxweight_stability_from_calibrated_fabric
    (M : FiniteSupportQueueModel I Ω)
    (full cand : ActionFamily A)
    (μtrue lower : A → ServiceVec I)
    (Φ : FabricFeature A D) (c : I → D → ℝ)
    (lam : ServiceVec I)
    (penalty : ServiceVec I → A → ℝ)
    (astar : NatQueueState I → A)
    (Amax Smax : ServiceVec I)
    {L ρ δ εest P0 β α : ℝ} {N : ℕ}
    (hAmax : Nonnegative Amax) (hSmax : Nonnegative Smax)
    (hArr_bound : ∀ Q : NatQueueState I, ∀ ω : Ω, ∀ i : I,
      (M.arrivals Q ω i : ℝ) ≤ Amax i)
    (hServ_bound : ∀ Q : NatQueueState I, ∀ ω : Ω, ∀ i : I,
      (M.service Q ω i : ℝ) ≤ Smax i)
    (hL : 0 ≤ L) (hρ : 0 ≤ ρ)
    (π : FabricCandidateProjection full cand Φ ρ)
    (henv : FeatureServiceSensitivityEnvelope μtrue Φ c)
    (hdom : FeatureSensitivityDominatedByFabricMetric Φ c L)
    (hcap : InCapacityWithSlack full μtrue lam δ)
    (hlower_gap : ∀ q : ServiceVec I, Nonnegative q →
      support cand μtrue q ≤ support cand lower q + εest * l1 q)
    (hmax : ∀ Q : NatQueueState I,
      RobustScoreMaximizer cand lower (natQueueToServiceVec Q)
        (penalty (natQueueToServiceVec Q)) (astar Q))
    (hpen : QueueScaledPenaltyBounded cand penalty P0 β)
    (harrival_coord : ∀ Q : NatQueueState I, ∀ i : I,
      M.expectedArrival Q i ≤ lam i)
    (hservice_coord : ∀ Q : NatQueueState I, ∀ i : I,
      lower (astar Q) i ≤ M.expectedService Q i)
    (hmargin : 0 < δ - (L * ρ + εest + β))
    (hα : 0 < α)
    (hN : secondOrderTerm Amax Smax + P0 + α
      ≤ (δ - (L * ρ + εest + β)) * (N : ℝ)) :
    PositiveRecurrentViaFiniteSet M.K (natQueueSmallSet (I := I) N) := by
  exact main_theorem_robust_candidate_maxweight_stability_under_fabric_cover
    M full cand μtrue lower Φ lam penalty astar Amax Smax
    hAmax hSmax hArr_bound hServ_bound hL hρ
    (π.covers full cand Φ ρ)
    (fabric_service_lipschitz_of_feature_sensitivity μtrue Φ c L henv hdom)
    hcap hlower_gap hmax hpen harrival_coord hservice_coord hmargin hα hN

/-- Calibrated paper main theorem with a bounded conditional second-order
moment.  This is the formal version closest to the standard queueing
literature statement; the bounded-sample theorem above is its constructive
finite-support specialization. -/
theorem main_theorem_robust_candidate_maxweight_stability_from_calibrated_fabric_with_second_moment_bound
    (M : FiniteSupportQueueModel I Ω)
    (full cand : ActionFamily A)
    (μtrue lower : A → ServiceVec I)
    (Φ : FabricFeature A D) (c : I → D → ℝ)
    (lam : ServiceVec I)
    (penalty : ServiceVec I → A → ℝ)
    (astar : NatQueueState I → A)
    {L ρ δ εest B P0 β α : ℝ} {N : ℕ}
    (hL : 0 ≤ L) (hρ : 0 ≤ ρ)
    (π : FabricCandidateProjection full cand Φ ρ)
    (henv : FeatureServiceSensitivityEnvelope μtrue Φ c)
    (hdom : FeatureSensitivityDominatedByFabricMetric Φ c L)
    (hcap : InCapacityWithSlack full μtrue lam δ)
    (hlower_gap : ∀ q : ServiceVec I, Nonnegative q →
      support cand μtrue q ≤ support cand lower q + εest * l1 q)
    (hmax : ∀ Q : NatQueueState I,
      RobustScoreMaximizer cand lower (natQueueToServiceVec Q)
        (penalty (natQueueToServiceVec Q)) (astar Q))
    (hpen : QueueScaledPenaltyBounded cand penalty P0 β)
    (hsecond : ∀ Q : NatQueueState I, M.expectedSecondOrder Q ≤ B)
    (harrival_coord : ∀ Q : NatQueueState I, ∀ i : I,
      M.expectedArrival Q i ≤ lam i)
    (hservice_coord : ∀ Q : NatQueueState I, ∀ i : I,
      lower (astar Q) i ≤ M.expectedService Q i)
    (hmargin : 0 < δ - (L * ρ + εest + β))
    (hα : 0 < α)
    (hN : B + P0 + α
      ≤ (δ - (L * ρ + εest + β)) * (N : ℝ)) :
    PositiveRecurrentViaFiniteSet M.K (natQueueSmallSet (I := I) N) := by
  exact main_theorem_robust_candidate_maxweight_stability_with_second_moment_bound
    M full cand μtrue lower Φ lam penalty astar hL hρ
    (π.covers full cand Φ ρ)
    (fabric_service_lipschitz_of_feature_sensitivity μtrue Φ c L henv hdom)
    hcap hlower_gap hmax hpen hsecond harrival_coord hservice_coord
    hmargin hα hN

/-- Calibrated bounded-second-moment theorem with a queue-scaled approximate
optimization oracle. -/
theorem main_theorem_robust_candidate_maxweight_stability_from_calibrated_fabric_with_second_moment_bound_approx_oracle
    (M : FiniteSupportQueueModel I Ω)
    (full cand : ActionFamily A)
    (μtrue lower : A → ServiceVec I)
    (Φ : FabricFeature A D) (c : I → D → ℝ)
    (lam : ServiceVec I)
    (penalty : ServiceVec I → A → ℝ)
    (astar : NatQueueState I → A)
    {L ρ δ εest B P0 β α0 α1 α : ℝ} {N : ℕ}
    (hL : 0 ≤ L) (hρ : 0 ≤ ρ)
    (π : FabricCandidateProjection full cand Φ ρ)
    (henv : FeatureServiceSensitivityEnvelope μtrue Φ c)
    (hdom : FeatureSensitivityDominatedByFabricMetric Φ c L)
    (hcap : InCapacityWithSlack full μtrue lam δ)
    (hlower_gap : ∀ q : ServiceVec I, Nonnegative q →
      support cand μtrue q ≤ support cand lower q + εest * l1 q)
    (hmax : ∀ Q : NatQueueState I,
      QueueScaledApproxRobustScoreMaximizer cand lower
        (natQueueToServiceVec Q) penalty (astar Q) α0 α1)
    (hpen : QueueScaledPenaltyBounded cand penalty P0 β)
    (hsecond : ∀ Q : NatQueueState I, M.expectedSecondOrder Q ≤ B)
    (harrival_coord : ∀ Q : NatQueueState I, ∀ i : I,
      M.expectedArrival Q i ≤ lam i)
    (hservice_coord : ∀ Q : NatQueueState I, ∀ i : I,
      lower (astar Q) i ≤ M.expectedService Q i)
    (hmargin : 0 < δ - (L * ρ + εest + β + α1))
    (hα : 0 < α)
    (hN : B + (P0 + α0) + α
      ≤ (δ - (L * ρ + εest + β + α1)) * (N : ℝ)) :
    PositiveRecurrentViaFiniteSet M.K (natQueueSmallSet (I := I) N) := by
  exact
    main_theorem_robust_candidate_maxweight_stability_with_second_moment_bound_approx_oracle
      M full cand μtrue lower Φ lam penalty astar hL hρ
      (π.covers full cand Φ ρ)
      (fabric_service_lipschitz_of_feature_sensitivity μtrue Φ c L henv hdom)
      hcap hlower_gap hmax hpen hsecond harrival_coord hservice_coord
      hmargin hα hN

/-- Statewise/dynamic-feasible-family version of the calibrated bounded
second-moment theorem with a queue-scaled approximate oracle.

This is the version closest to the real scheduler boundary: the full feasible
action family, maintained candidate family, service map, lower-service map,
fabric features, and penalty may depend on the current queue/state snapshot.
The theorem requires the calibrated fabric projection and the support/slack
conditions to hold at every state. -/
theorem main_statewise_calibrated_fabric_robust_candidate_stability_with_second_moment_bound_approx_oracle
    (M : FiniteSupportQueueModel I Ω)
    (full cand : NatQueueState I → ActionFamily A)
    (μtrue lower : NatQueueState I → A → ServiceVec I)
    (Φ : NatQueueState I → FabricFeature A D)
    (c : NatQueueState I → I → D → ℝ)
    (lam : ServiceVec I)
    (penalty : NatQueueState I → ServiceVec I → A → ℝ)
    (astar : NatQueueState I → A)
    {L ρ δ εest B P0 β α0 α1 α : ℝ} {N : ℕ}
    (hL : 0 ≤ L) (hρ : 0 ≤ ρ)
    (π : ∀ Q : NatQueueState I,
      FabricCandidateProjection (full Q) (cand Q) (Φ Q) ρ)
    (henv : ∀ Q : NatQueueState I,
      FeatureServiceSensitivityEnvelope (μtrue Q) (Φ Q) (c Q))
    (hdom : ∀ Q : NatQueueState I,
      FeatureSensitivityDominatedByFabricMetric (Φ Q) (c Q) L)
    (hcap : ∀ Q : NatQueueState I,
      InCapacityWithSlack (full Q) (μtrue Q) lam δ)
    (hlower_gap : ∀ Q : NatQueueState I, ∀ q : ServiceVec I,
      Nonnegative q →
        support (cand Q) (μtrue Q) q
          ≤ support (cand Q) (lower Q) q + εest * l1 q)
    (hmax : ∀ Q : NatQueueState I,
      QueueScaledApproxRobustScoreMaximizer (cand Q) (lower Q)
        (natQueueToServiceVec Q) (penalty Q) (astar Q) α0 α1)
    (hpen : ∀ Q : NatQueueState I,
      QueueScaledPenaltyBounded (cand Q) (penalty Q) P0 β)
    (hsecond : ∀ Q : NatQueueState I, M.expectedSecondOrder Q ≤ B)
    (harrival_coord : ∀ Q : NatQueueState I, ∀ i : I,
      M.expectedArrival Q i ≤ lam i)
    (hservice_coord : ∀ Q : NatQueueState I, ∀ i : I,
      (lower Q (astar Q)) i ≤ M.expectedService Q i)
    (hmargin : 0 < δ - (L * ρ + εest + β + α1))
    (hα : 0 < α)
    (hN : B + (P0 + α0) + α
      ≤ (δ - (L * ρ + εest + β + α1)) * (N : ℝ)) :
    PositiveRecurrentViaFiniteSet M.K (natQueueSmallSet (I := I) N) := by
  let selectedLower : NatQueueState I → ServiceVec I :=
    fun Q => lower Q (astar Q)
  have hlinear : ∀ Q : NatQueueState I,
      B + dot (natQueueToServiceVec Q) lam
          - dot (natQueueToServiceVec Q) (selectedLower Q)
        ≤ B + (P0 + α0)
            - (δ - (L * ρ + εest + β + α1)) * natQueueL1 Q := by
    intro Q
    let q : ServiceVec I := natQueueToServiceVec Q
    have hq : Nonnegative q := natQueueToServiceVec_nonnegative Q
    have hgap : SupportGapAtMost (full Q) (cand Q) (μtrue Q) (L * ρ) :=
      calibrated_fabric_cover_support_gap
        (full Q) (cand Q) (μtrue Q) (Φ Q) (c Q)
        hL hρ (π Q) (henv Q) (hdom Q)
    have hslack : dot q lam + δ * l1 q ≤ support (full Q) (μtrue Q) q :=
      capacity_slack_implies_support_slack (full Q) (μtrue Q) hq (hcap Q)
    have happrox : support (full Q) (μtrue Q) q
        ≤ dot q (lower Q (astar Q))
            + (L * ρ + εest + β + α1) * l1 q + (P0 + α0) :=
      robust_candidate_policy_approx_full_support_scaled_penalty_approx_oracle
        (full Q) (cand Q) (μtrue Q) (lower Q)
        hq hgap (hlower_gap Q q hq) (penalty Q) (hmax Q) (hpen Q)
    have hl1 : l1 q = natQueueL1 Q := l1_natQueueToServiceVec Q
    unfold selectedLower
    change B + dot q lam - dot q (lower Q (astar Q))
      ≤ B + (P0 + α0)
          - (δ - (L * ρ + εest + β + α1)) * natQueueL1 Q
    rw [← hl1]
    linarith
  exact M.positive_recurrent_via_coordinate_moments'
    (fun _ : NatQueueState I => B)
    (fun _ : NatQueueState I => lam)
    selectedLower
    hsecond
    harrival_coord
    hservice_coord
    hmargin hα hN hlinear

/-- Hidden-regime extension: dwell-time/switching windows reduce the drift
margin by the explicit factor `χθ`, rather than being hidden in a vague
nonstationarity penalty. -/
theorem main_hidden_regime_dwell_switching_drift
    (drift : T → ℝ) (Q : T → ServiceVec I)
    (segmentOf : T → K) (marked : T → Prop) [DecidablePred marked]
    (fixed : K → ℝ) (estimationLoss : T → ℝ)
    {B margin χ θ switchingCost : ℝ}
    (hχ : 0 ≤ χ)
    (hstep : ∀ t : T,
      drift t ≤ B - margin * l1 (Q t) + estimationLoss t)
    (hseg : ∀ k : K,
      markedBacklogBySegment segmentOf marked Q k
        ≤ θ * segmentBacklog segmentOf Q k + fixed k) :
    (∑ t : T, drift t) + χ * markedBacklog marked Q + switchingCost
      ≤
    (Fintype.card T : ℝ) * B
      - (margin - χ * θ) * (∑ t : T, l1 (Q t))
      + estimationPenalty estimationLoss
      + χ * (∑ k : K, fixed k)
      + switchingCost := by
  exact cumulative_drift_with_dwell_switching_budget
    drift Q segmentOf marked fixed estimationLoss hχ hstep hseg

/-- Structured-learning extension: learning regret scales with active
fabric/co-location buckets, not with the full global action count. -/
theorem main_active_bucket_lcb_learning_regret
    (active : Finset Bkt)
    (F : ActionFamily A)
    (μtrue μhat : T → A → ServiceVec I)
    (rad : T → A → ℝ)
    (Q : T → ServiceVec I) (ε : T → ℝ)
    (bucketLoss bucketCount : Bkt → ℝ)
    {Qmax C horizonMass : ℝ}
    (hQ : ∀ t : T, Nonnegative (Q t))
    (hε : ∀ t : T, 0 ≤ ε t)
    (hpath : LCBConfidencePathEvent F μtrue μhat rad ε)
    (hbucket :
      ActiveBucketConcentrationEvent active ε bucketLoss bucketCount
        C horizonMass)
    (hQmax : ∀ t : T, l1 (Q t) ≤ Qmax)
    (hQmax_nonneg : 0 ≤ Qmax) :
    (∑ t : T,
      (support F (μtrue t) (Q t)
        - support F (lcbService (μhat t) (rad t)) (Q t)))
      ≤
    2 * Qmax * C * Real.sqrt ((active.card : ℝ) * horizonMass) := by
  exact lcb_learning_regret_from_active_bucket_concentration_event
    active F μtrue μhat rad Q ε bucketLoss bucketCount
    hQ hε hpath hbucket hQmax hQmax_nonneg

/-- High-probability structured-learning extension.  If the concrete adaptive
sampler proves the active-bucket input event with probability at least `p`,
then the non-vacuous active-bucket regret bound also holds with probability at
least `p`. -/
theorem main_active_bucket_lcb_learning_regret_high_probability
    {Ωp : Type*} [MeasurableSpace Ωp]
    (ℙ : MeasureTheory.Measure Ωp) {p : ENNReal}
    (active : Finset Bkt)
    (F : ActionFamily A)
    (μtrue μhat : Ωp → T → A → ServiceVec I)
    (rad : Ωp → T → A → ℝ)
    (Q : Ωp → T → ServiceVec I) (ε : Ωp → T → ℝ)
    (bucketLoss bucketCount : Ωp → Bkt → ℝ)
    (Qmax C horizonMass : Ωp → ℝ)
    (hprob :
      EventProbabilityAtLeast ℙ
        (fun ω =>
          ActiveBucketLCBInputEvent active F (μtrue ω) (μhat ω)
            (rad ω) (Q ω) (ε ω) (bucketLoss ω) (bucketCount ω)
            (Qmax ω) (C ω) (horizonMass ω)) p) :
    EventProbabilityAtLeast ℙ
      (fun ω =>
        ActiveBucketLCBRegretBoundEvent active F (μtrue ω) (μhat ω)
          (rad ω) (Q ω) (ε ω) (Qmax ω) (C ω) (horizonMass ω)) p := by
  exact active_bucket_lcb_learning_regret_high_probability
    ℙ active F μtrue μhat rad Q ε bucketLoss bucketCount Qmax C
    horizonMass hprob

/-- Generic high-probability stability lifting.  A learning or posterior
analysis may prove a certificate event, for example lower-service domination
for the selected actions.  If that event implies the deterministic Foster
stability certificate, then the stability certificate holds with the same
probability.  A sampler-specific theorem must still prove the input event. -/
theorem main_high_probability_stability_from_certificate_event
    {Ωp : Type*} [MeasurableSpace Ωp]
    (ℙ : MeasureTheory.Measure Ωp) {p : ENNReal}
    (M : FiniteSupportQueueModel I Ω) {N : ℕ}
    (certificate : Ωp → Prop)
    (hprob : EventProbabilityAtLeast ℙ certificate p)
    (hcert : ∀ ω,
      certificate ω →
        PositiveRecurrentViaFiniteSet M.K (natQueueSmallSet (I := I) N)) :
    EventProbabilityAtLeast ℙ
      (fun _ =>
        PositiveRecurrentViaFiniteSet M.K (natQueueSmallSet (I := I) N)) p := by
  exact event_probability_mono ℙ hprob hcert

/-- Active-bucket local concentration union bound.  This theorem supplies the
finite active-bucket probability aggregation that turns per-bucket local
confidence controls into an all-active-bucket confidence event. -/
theorem main_active_bucket_local_failure_union_bound
    {Ωp : Type*} [MeasurableSpace Ωp]
    (ℙ : MeasureTheory.Measure Ωp)
    (active : Finset Bkt)
    (E : Bkt → Ωp → Prop) (δ : Bkt → ENNReal)
    (hfail : ∀ b ∈ active,
      EventFailureProbabilityAtMost ℙ (E b) (δ b)) :
    EventFailureProbabilityAtMost ℙ
      (fun ω => ∀ b ∈ active, E b ω)
      (∑ b ∈ active, δ b) := by
  exact active_finset_all_events_failure_probability ℙ active E δ hfail

end Scheduleurm


/-! ## Source: Scheduleurm/FrameBasedStability.lean -/


/-!
# Variable-duration frame stability for Scheduleurm

This file closes the event-driven boundary used by migration and port actions.
A frame has a strictly positive physical duration `τ`, cumulative arrivals
`τ λ`, and an arbitrary cumulative lower-service vector `ServCum`.  The robust
oracle is charged in cumulative units, including bounded and queue-scaled
optimization and reconfiguration losses.

The main drift theorem proves

`ΔV ≤ B + α₀ + P₀ - τ (δ - ε - α₁ - β) ||Q||₁`.

Thus a uniform lower frame duration transfers a positive per-unit-time slack
to a uniform embedded-chain Foster drift.  A separate upper duration bound
controls elapsed physical time.  The duration-one corollary reduces exactly to
the ordinary slotted queue update; no uniformization or zero-duration action is
silently assumed.
-/

noncomputable section

set_option linter.unusedSectionVars false
set_option linter.unusedVariables false

namespace Scheduleurm

open BigOperators

variable {I A : Type*} [Fintype I] [DecidableEq I] [DecidableEq A]

/-- Pointwise scalar multiplication for arrival and service vectors. -/
def scaleService (c : ℝ) (v : ServiceVec I) : ServiceVec I :=
  fun i => c * v i

/-- Cumulative arrivals during a frame of duration `τ`. -/
def frameArrivals (τ : ℝ) (lam : ServiceVec I) : ServiceVec I :=
  scaleService τ lam

/-- Queue update at a frame boundary. -/
def frameQueueStep
    (Q lam ServCum : ServiceVec I) (τ : ℝ) : ServiceVec I :=
  queueStep Q (frameArrivals τ lam) ServCum

/-- Duration-normalized cumulative service.  It is used only when `τ > 0`. -/
def normalizedCumulativeService (τ : ℝ) (ServCum : ServiceVec I) : ServiceVec I :=
  fun i => ServCum i / τ

lemma scaleService_nonnegative
    {c : ℝ} {v : ServiceVec I}
    (hc : 0 ≤ c) (hv : Nonnegative v) :
    Nonnegative (scaleService c v) := by
  intro i
  exact mul_nonneg hc (hv i)

lemma dot_scaleService (q v : ServiceVec I) (c : ℝ) :
    dot q (scaleService c v) = c * dot q v := by
  unfold dot scaleService
  calc
    ∑ i : I, q i * (c * v i) = ∑ i : I, c * (q i * v i) := by
      apply Finset.sum_congr rfl
      intro i _
      ring
    _ = c * ∑ i : I, q i * v i := by
      rw [Finset.mul_sum]

lemma dot_normalizedCumulativeService
    (q ServCum : ServiceVec I) (τ : ℝ) :
    dot q (normalizedCumulativeService τ ServCum) = dot q ServCum / τ := by
  unfold dot normalizedCumulativeService
  calc
    ∑ i : I, q i * (ServCum i / τ) = ∑ i : I, (q i * ServCum i) / τ := by
      apply Finset.sum_congr rfl
      intro i _
      ring
    _ = (∑ i : I, q i * ServCum i) / τ := by
      rw [Finset.sum_div]

lemma secondOrderTerm_scaleService
    (Arr Serv : ServiceVec I) (τ : ℝ) :
    secondOrderTerm (scaleService τ Arr) (scaleService τ Serv)
      = τ^2 * secondOrderTerm Arr Serv := by
  unfold secondOrderTerm scaleService
  calc
    ∑ i : I, (((τ * Arr i)^2 + (τ * Serv i)^2) / 2)
        = ∑ i : I, τ^2 * (((Arr i)^2 + (Serv i)^2) / 2) := by
          apply Finset.sum_congr rfl
          intro i _
          ring
    _ = τ^2 * ∑ i : I, (((Arr i)^2 + (Serv i)^2) / 2) := by
          rw [Finset.mul_sum]

/-- A rate-level second-order bound becomes a frame bound under `τ ≤ τmax`. -/
theorem secondOrderTerm_frame_le
    {Arr Serv : ServiceVec I} {τ τmax Bunit : ℝ}
    (hτ : 0 ≤ τ) (hupper : τ ≤ τmax) (hBunit : 0 ≤ Bunit)
    (hsecond : secondOrderTerm Arr Serv ≤ Bunit) :
    secondOrderTerm (scaleService τ Arr) (scaleService τ Serv)
      ≤ τmax^2 * Bunit := by
  rw [secondOrderTerm_scaleService]
  have hτmax : 0 ≤ τmax := le_trans hτ hupper
  have hsquares : τ^2 ≤ τmax^2 := by nlinarith
  have hterm : 0 ≤ secondOrderTerm Arr Serv := by
    unfold secondOrderTerm
    positivity
  calc
    τ^2 * secondOrderTerm Arr Serv
        ≤ τmax^2 * secondOrderTerm Arr Serv :=
          mul_le_mul_of_nonneg_right hsquares hterm
    _ ≤ τmax^2 * Bunit :=
          mul_le_mul_of_nonneg_left hsecond (sq_nonneg τmax)

/-- Cumulative robust-oracle obligation over one frame. -/
def FrameApproximateOracle
    (full : ActionFamily A) (μ : A → ServiceVec I)
    (Q ServCum : ServiceVec I)
    (τ ε α0 α1 penalty : ℝ) : Prop :=
  τ * support full μ Q
    ≤ dot Q ServCum
      + τ * ε * l1 Q
      + α0
      + τ * α1 * l1 Q
      + penalty

/-- Reconfiguration, migration and risk loss with bounded and linear parts. -/
def FramePenaltyBound
    (Q : ServiceVec I) (τ penalty P0 β : ℝ) : Prop :=
  penalty ≤ P0 + τ * β * l1 Q

/-- The cumulative oracle inequality is exactly the duration-normalized
inequality when `τ > 0`. -/
theorem frameApproximateOracle_iff_durationNormalized
    (full : ActionFamily A) (μ : A → ServiceVec I)
    (Q ServCum : ServiceVec I)
    {τ ε α0 α1 penalty : ℝ} (hτ : 0 < τ) :
    FrameApproximateOracle full μ Q ServCum τ ε α0 α1 penalty ↔
      support full μ Q
        ≤ (dot Q ServCum
            + τ * ε * l1 Q
            + α0
            + τ * α1 * l1 Q
            + penalty) / τ := by
  unfold FrameApproximateOracle
  simpa [mul_comm] using (le_div_iff₀ hτ).symm

/-- Variable-duration robust MaxWeight pressure inequality. -/
theorem frame_approximate_maxWeight_negative_drift
    (full : ActionFamily A) (μ : A → ServiceVec I)
    {Q lam ServCum : ServiceVec I}
    {τ δ ε α0 α1 penalty P0 β B : ℝ}
    (hτ : 0 < τ)
    (hQ : Nonnegative Q)
    (hcap : InCapacityWithSlack full μ lam δ)
    (horacle : FrameApproximateOracle
      full μ Q ServCum τ ε α0 α1 penalty)
    (hpenalty : FramePenaltyBound Q τ penalty P0 β)
    (hSecond : secondOrderTerm (frameArrivals τ lam) ServCum ≤ B) :
    secondOrderTerm (frameArrivals τ lam) ServCum
        + dot Q (frameArrivals τ lam) - dot Q ServCum
      ≤ B + α0 + P0
        - τ * (δ - ε - α1 - β) * l1 Q := by
  have hSlack : dot Q lam + δ * l1 Q ≤ support full μ Q :=
    capacity_slack_implies_support_slack full μ hQ hcap
  have hτnonneg : 0 ≤ τ := le_of_lt hτ
  have hScaledSlack :
      τ * (dot Q lam + δ * l1 Q) ≤ τ * support full μ Q :=
    mul_le_mul_of_nonneg_left hSlack hτnonneg
  have hOracleExpanded := horacle
  unfold FrameApproximateOracle at hOracleExpanded
  have hPenaltyExpanded := hpenalty
  unfold FramePenaltyBound at hPenaltyExpanded
  rw [frameArrivals, dot_scaleService] at *
  nlinarith

/-- Full variable-duration frame Lyapunov drift theorem. -/
theorem frame_approximate_maxWeight_lyapunov_drift
    (full : ActionFamily A) (μ : A → ServiceVec I)
    {Q lam ServCum : ServiceVec I}
    {τ δ ε α0 α1 penalty P0 β B : ℝ}
    (hτ : 0 < τ)
    (hQ : Nonnegative Q) (hlam : Nonnegative lam)
    (hServCum : Nonnegative ServCum)
    (hcap : InCapacityWithSlack full μ lam δ)
    (horacle : FrameApproximateOracle
      full μ Q ServCum τ ε α0 α1 penalty)
    (hpenalty : FramePenaltyBound Q τ penalty P0 β)
    (hSecond : secondOrderTerm (frameArrivals τ lam) ServCum ≤ B) :
    Lyapunov (frameQueueStep Q lam ServCum τ) - Lyapunov Q
      ≤ B + α0 + P0
        - τ * (δ - ε - α1 - β) * l1 Q := by
  have hArr : Nonnegative (frameArrivals τ lam) := by
    unfold frameArrivals
    exact scaleService_nonnegative (le_of_lt hτ) hlam
  have hQueue := lyapunov_queue_step_bound
    (Q := Q) (Arr := frameArrivals τ lam) (Serv := ServCum)
    hQ hArr hServCum
  have hPressure := frame_approximate_maxWeight_negative_drift
    full μ hτ hQ hcap horacle hpenalty hSecond
  unfold frameQueueStep
  exact le_trans hQueue hPressure

/-- A positive per-time slack and a positive minimum frame duration give a
uniform embedded-chain Foster coefficient. -/
theorem frame_approximate_maxWeight_lyapunov_drift_uniform
    (full : ActionFamily A) (μ : A → ServiceVec I)
    {Q lam ServCum : ServiceVec I}
    {τ τmin δ ε α0 α1 penalty P0 β B : ℝ}
    (hτ : 0 < τ) (hτmin : 0 < τmin) (hlower : τmin ≤ τ)
    (hmargin : 0 < δ - ε - α1 - β)
    (hQ : Nonnegative Q) (hlam : Nonnegative lam)
    (hServCum : Nonnegative ServCum)
    (hcap : InCapacityWithSlack full μ lam δ)
    (horacle : FrameApproximateOracle
      full μ Q ServCum τ ε α0 α1 penalty)
    (hpenalty : FramePenaltyBound Q τ penalty P0 β)
    (hSecond : secondOrderTerm (frameArrivals τ lam) ServCum ≤ B) :
    Lyapunov (frameQueueStep Q lam ServCum τ) - Lyapunov Q
      ≤ B + α0 + P0
        - τmin * (δ - ε - α1 - β) * l1 Q := by
  have hdrift := frame_approximate_maxWeight_lyapunov_drift
    full μ hτ hQ hlam hServCum hcap horacle hpenalty hSecond
  have hbacklog := l1_nonneg Q
  have hscaled :
      τmin * (δ - ε - α1 - β) * l1 Q
        ≤ τ * (δ - ε - α1 - β) * l1 Q := by
    have hnonneg : 0 ≤ (δ - ε - α1 - β) * l1 Q :=
      mul_nonneg (le_of_lt hmargin) hbacklog
    simpa [mul_assoc] using mul_le_mul_of_nonneg_right hlower hnonneg
  linarith

/-- Duration one recovers the ordinary slotted queue theorem with all error
and penalty terms retained. -/
theorem frame_lyapunov_drift_duration_one
    (full : ActionFamily A) (μ : A → ServiceVec I)
    {Q lam Serv : ServiceVec I}
    {δ ε α0 α1 penalty P0 β B : ℝ}
    (hQ : Nonnegative Q) (hlam : Nonnegative lam)
    (hServ : Nonnegative Serv)
    (hcap : InCapacityWithSlack full μ lam δ)
    (horacle : support full μ Q
      ≤ dot Q Serv + ε * l1 Q + α0 + α1 * l1 Q + penalty)
    (hpenalty : penalty ≤ P0 + β * l1 Q)
    (hSecond : secondOrderTerm lam Serv ≤ B) :
    Lyapunov (queueStep Q lam Serv) - Lyapunov Q
      ≤ B + α0 + P0 - (δ - ε - α1 - β) * l1 Q := by
  have hFrameOracle : FrameApproximateOracle
      full μ Q Serv 1 ε α0 α1 penalty := by
    unfold FrameApproximateOracle
    simpa [one_mul, add_assoc] using horacle
  have hFramePenalty : FramePenaltyBound Q 1 penalty P0 β := by
    unfold FramePenaltyBound
    simpa using hpenalty
  have hSecondFrame : secondOrderTerm (frameArrivals 1 lam) Serv ≤ B := by
    simpa [frameArrivals, scaleService, secondOrderTerm] using hSecond
  have hdrift := frame_approximate_maxWeight_lyapunov_drift
    full μ (τ := 1) (B := B) (by norm_num) hQ hlam hServ hcap
    hFrameOracle hFramePenalty hSecondFrame
  have hscale : scaleService 1 lam = lam := by
    funext i
    simp [scaleService]
  unfold frameQueueStep frameArrivals at hdrift
  rw [hscale] at hdrift
  simpa using hdrift

/-- Elapsed physical time over the first `n` frames. -/
def elapsedFrameTime (duration : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ k ∈ Finset.range n, duration k

/-- Uniformly bounded frame durations transfer frame-count bounds to physical
time bounds. -/
theorem elapsedFrameTime_le
    (duration : ℕ → ℝ) {τmax : ℝ}
    (hupper : ∀ k, duration k ≤ τmax) (n : ℕ) :
    elapsedFrameTime duration n ≤ (n : ℝ) * τmax := by
  unfold elapsedFrameTime
  calc
    ∑ k ∈ Finset.range n, duration k
        ≤ ∑ k ∈ Finset.range n, τmax := by
          apply Finset.sum_le_sum
          intro k hk
          exact hupper k
    _ = (n : ℝ) * τmax := by simp

/-- Embedded frame-boundary positive recurrence under the uniform frame drift.
The calendar-time interpretation additionally uses `elapsedFrameTime_le`. -/
theorem frame_nat_model_positive_recurrent_via_finite_set
    (M : NatQueueTransitionModel I)
    {B α0 P0 τmin margin α : ℝ} {N : ℕ}
    (hτmin : 0 < τmin) (hmargin : 0 < margin) (hα : 0 < α)
    (hN : B + α0 + P0 + α ≤ τmin * margin * (N : ℝ))
    (hdet : ∀ Q : NatQueueState I,
      M.deterministicDrift Q
        ≤ B + α0 + P0 - τmin * margin * natQueueL1 Q)
    (hreturn : ∀ Q : NatQueueState I,
      natQueueSmallSet (I := I) N Q →
        FiniteExpectedReturnTimeToSet M.K (natQueueSmallSet (I := I) N) Q) :
    PositiveRecurrentViaFiniteSet M.K (natQueueSmallSet (I := I) N) := by
  have hη : 0 < τmin * margin := mul_pos hτmin hmargin
  apply nat_model_linear_drift_positive_recurrent_via_finite_set
    M (B := B + α0 + P0) (η := τmin * margin) (α := α) (N := N)
    hη hα
  · nlinarith
  · intro Q
    have h := hdet Q
    nlinarith
  · exact hreturn


/-! ## Source: Scheduleurm/ActionUnion.lean -/

/-!
# Scheduleurm: finite candidate action unions

This section formalizes the monotonicity used when a feasible external solver
or policy trajectory is admitted into an already certified finite candidate
family. Candidate expansion cannot worsen the certified support loss, so the
expanded family inherits the same capacity-slack guarantee.
-/

/-- Enlarging a finite candidate family cannot worsen any previously certified
support-function loss. -/
theorem support_gap_mono_under_candidate_expansion
    (full base expanded : ActionFamily A) (mu : A -> ServiceVec I) {epsilon : Real}
    (hsubset : ActionFamily.Subset base expanded)
    (hgap : SupportGapAtMost full base mu epsilon) :
    SupportGapAtMost full expanded mu epsilon := by
  intro q hq
  exact le_trans (hgap q hq)
    (by
      simpa [add_comm] using
        (add_le_add_right
          (support_mono base expanded mu q hsubset) (epsilon * l1 q)))

/-- A certified candidate-family expansion inherits the base family's
full-action support loss and therefore preserves the same capacity-slack lower
bound. -/
theorem candidate_capacity_slack_loss_under_expansion
    (full base expanded : ActionFamily A) (mu : A -> ServiceVec I)
    {lam q : ServiceVec I} {delta epsilon : Real}
    (hq : Nonnegative q)
    (hcap : InCapacityWithSlack full mu lam delta)
    (hsubset : ActionFamily.Subset base expanded)
    (hgap : SupportGapAtMost full base mu epsilon) :
    dot q lam + (delta - epsilon) * l1 q <= support expanded mu q := by
  exact candidate_capacity_slack_loss full expanded mu hq hcap
    (support_gap_mono_under_candidate_expansion
      full base expanded mu hsubset hgap)

end Scheduleurm
