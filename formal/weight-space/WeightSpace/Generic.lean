import WeightSpace.Bounds

/-!
# Generic frozen-feature weight programs

The feature map is arbitrary and fixed: it may be a neural backbone, a nonlinear
feature extractor, or a hand-written map. Inputs and outputs need not describe
games. Bias is represented by an explicit constant feature when needed.

Every equality and inequality is over the real numbers. These theorems do not
assert that a numerical SVD produces an exact nullspace, or that floating-point
execution refines real arithmetic. Endpoint residuals, not solver status, are
the premises of the protection statements.
-/

noncomputable section

open scoped BigOperators

namespace WeightSpace.Generic

variable {X F O A R : Type*} [Fintype F]

/-- A general multi-output linear head over an arbitrary frozen feature map. -/
def readout (w : O → F → ℝ) (φ : X → F → ℝ) (x : X) (o : O) : ℝ :=
  ∑ f, w o f * φ x f

def edit (w d : O → F → ℝ) : O → F → ℝ := fun o f => w o f + d o f

/-- Actual endpoint differences are distinguished from requested edits. -/
def endpointDelta (w q : O → F → ℝ) : O → F → ℝ := fun o f => q o f - w o f

theorem edit_exact (w d : O → F → ℝ) (φ : X → F → ℝ) (x : X) (o : O) :
    readout (edit w d) φ x o - readout w φ x o = readout d φ x o := by
  simp only [readout, edit, add_mul, Finset.sum_add_distrib]
  ring

theorem endpoint_exact (w q : O → F → ℝ) (φ : X → F → ℝ) (x : X) (o : O) :
    readout q φ x o - readout w φ x o = readout (endpointDelta w q) φ x o := by
  simp only [readout, endpointDelta, sub_mul, Finset.sum_sub_distrib]

theorem effect_add (d e : O → F → ℝ) (φ : X → F → ℝ) (x : X) (o : O) :
    readout (edit d e) φ x o = readout d φ x o + readout e φ x o := by
  simp only [readout, edit, add_mul, Finset.sum_add_distrib]

omit [Fintype F] in
theorem edit_compose (w d e : O → F → ℝ) : edit (edit w d) e = edit w (edit d e) := by
  funext o f
  exact add_assoc _ _ _

omit [Fintype F] in
theorem edit_rollback (w d : O → F → ℝ) :
    edit (edit w d) (fun o f => -d o f) = w := by
  funext o f
  simp [edit]

/-- Finite-span membership with explicit witnesses; no numerical rank premise. -/
def InFeatureSpan [Fintype A] (φ : X → F → ℝ) (anchor : A → X)
    (c : A → ℝ) (x : X) : Prop :=
  ∀ f, φ x f = ∑ a, c a * φ (anchor a) f

/-- A linear readout evaluates finite feature combinations linearly. -/
theorem feature_span_effect [Fintype A] (d : O → F → ℝ) (φ : X → F → ℝ)
    (anchor : A → X) (c : A → ℝ) (x : X) (o : O)
    (hspan : InFeatureSpan φ anchor c x) :
    readout d φ x o = ∑ a, c a * readout d φ (anchor a) o := by
  unfold readout
  calc
    ∑ f, d o f * φ x f = ∑ f, ∑ a, c a * (d o f * φ (anchor a) f) := by
      apply Finset.sum_congr rfl
      intro f _
      rw [hspan f, Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro a _
      ring
    _ = ∑ a, ∑ f, c a * (d o f * φ (anchor a) f) := Finset.sum_comm
    _ = ∑ a, c a * ∑ f, d o f * φ (anchor a) f := by
      simp only [Finset.mul_sum]

/-- Zero endpoint residuals protect the entire represented feature span. -/
theorem protected_span [Fintype A] (w q : O → F → ℝ) (φ : X → F → ℝ)
    (anchor : A → X) (c : A → ℝ) (x : X) (o : O)
    (hspan : InFeatureSpan φ anchor c x)
    (hanchor : ∀ a, readout q φ (anchor a) o = readout w φ (anchor a) o) :
    readout q φ x o = readout w φ x o := by
  apply sub_eq_zero.mp
  rw [endpoint_exact, feature_span_effect (endpointDelta w q) φ anchor c x o hspan]
  apply Finset.sum_eq_zero
  intro a _
  rw [← endpoint_exact, hanchor a, sub_self, mul_zero]

theorem protected_span_edit [Fintype A] (w d : O → F → ℝ) (φ : X → F → ℝ)
    (anchor : A → X) (c : A → ℝ) (x : X) (o : O)
    (hspan : InFeatureSpan φ anchor c x)
    (hanchor : ∀ a, readout d φ (anchor a) o = 0) :
    readout (edit w d) φ x o = readout w φ x o := by
  apply sub_eq_zero.mp
  rw [edit_exact, feature_span_effect d φ anchor c x o hspan]
  simp only [hanchor, mul_zero, Finset.sum_const_zero]

/-- Coordinate-dependent anchor residual bounds retain coefficient conditioning. -/
theorem residual_span_bound [Fintype A] (d : O → F → ℝ) (φ : X → F → ℝ)
    (anchor : A → X) (c ε : A → ℝ) (x : X) (o : O)
    (hspan : InFeatureSpan φ anchor c x)
    (hanchor : ∀ a, |readout d φ (anchor a) o| ≤ ε a) :
    |readout d φ x o| ≤ ∑ a, |c a| * ε a := by
  rw [feature_span_effect d φ anchor c x o hspan]
  calc
    |∑ a, c a * readout d φ (anchor a) o|
        ≤ ∑ a, |c a * readout d φ (anchor a) o| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ a, |c a| * ε a := by
      apply Finset.sum_le_sum
      intro a _
      rw [abs_mul]
      exact mul_le_mul_of_nonneg_left (hanchor a) (abs_nonneg _)

/-- Uniform anchor residual ε is amplified by the explicit coefficient L1 norm. -/
theorem residual_span_l1_bound [Fintype A] (d : O → F → ℝ) (φ : X → F → ℝ)
    (anchor : A → X) (c : A → ℝ) (ε : ℝ) (x : X) (o : O)
    (hspan : InFeatureSpan φ anchor c x)
    (hanchor : ∀ a, |readout d φ (anchor a) o| ≤ ε) :
    |readout d φ x o| ≤ (∑ a, |c a|) * ε := by
  simpa only [Finset.sum_mul] using
    residual_span_bound d φ anchor c (fun _ => ε) x o hspan hanchor

/-- The bound is stated directly for actual endpoints, independently of solver. -/
theorem endpoint_span_bound [Fintype A] (w q : O → F → ℝ) (φ : X → F → ℝ)
    (anchor : A → X) (c ε : A → ℝ) (x : X) (o : O)
    (hspan : InFeatureSpan φ anchor c x)
    (hanchor : ∀ a, |readout q φ (anchor a) o - readout w φ (anchor a) o| ≤ ε a) :
    |readout q φ x o - readout w φ x o| ≤ ∑ a, |c a| * ε a := by
  rw [endpoint_exact]
  apply residual_span_bound (endpointDelta w q) φ anchor c ε x o hspan
  intro a
  simpa only [← endpoint_exact] using hanchor a

/-- Off-span feature error contributes a separate weighted residual term. -/
theorem approximate_span_bound [Fintype A] (d : O → F → ℝ) (φ : X → F → ℝ)
    (anchor : A → X) (c ε : A → ℝ) (ρ : F → ℝ) (x : X) (o : O)
    (hfeature : ∀ f, |φ x f - ∑ a, c a * φ (anchor a) f| ≤ ρ f)
    (hanchor : ∀ a, |readout d φ (anchor a) o| ≤ ε a) :
    |readout d φ x o| ≤ (∑ a, |c a| * ε a) + ∑ f, |d o f| * ρ f := by
  have hid : readout d φ x o = (∑ a, c a * readout d φ (anchor a) o) +
      ∑ f, d o f * (φ x f - ∑ a, c a * φ (anchor a) f) := by
    have hswap : (∑ a, c a * readout d φ (anchor a) o) =
        ∑ f, d o f * ∑ a, c a * φ (anchor a) f := by
      simp only [readout, Finset.mul_sum]
      rw [Finset.sum_comm]
      apply Finset.sum_congr rfl
      intro f _
      apply Finset.sum_congr rfl
      intro a _
      ring
    rw [hswap]
    simp only [readout, mul_sub, Finset.sum_sub_distrib]
    ring
  rw [hid]
  apply (abs_add_le _ _).trans
  apply add_le_add
  · apply (Finset.abs_sum_le_sum_abs _ _).trans
    apply Finset.sum_le_sum
    intro a _
    rw [abs_mul]
    exact mul_le_mul_of_nonneg_left (hanchor a) (abs_nonneg _)
  · apply (Finset.abs_sum_le_sum_abs _ _).trans
    apply Finset.sum_le_sum
    intro f _
    rw [abs_mul]
    exact mul_le_mul_of_nonneg_left (hfeature f) (abs_nonneg _)

/-- Exact protection makes a nonzero target change in that feature span infeasible. -/
theorem nonzero_target_in_protected_span_impossible [Fintype A]
    (w q : O → F → ℝ) (φ : X → F → ℝ) (anchor : A → X) (c : A → ℝ)
    (x : X) (o : O) (τ : ℝ) (hτ : τ ≠ 0)
    (hspan : InFeatureSpan φ anchor c x)
    (hanchor : ∀ a, readout q φ (anchor a) o = readout w φ (anchor a) o) :
    readout q φ x o - readout w φ x o ≠ τ := by
  rw [protected_span w q φ anchor c x o hspan hanchor, sub_self]
  exact Ne.symm hτ

/-- A factored program; the factor count is the type R. -/
def factoredEdit [Fintype R] (u : O → R → ℝ) (v : R → F → ℝ) : O → F → ℝ :=
  fun o f => ∑ r, u o r * v r f

/-- Factored effects are evaluated without materializing the full edit matrix. -/
theorem factored_effect [Fintype R] (u : O → R → ℝ) (v : R → F → ℝ)
    (φ : X → F → ℝ) (x : X) (o : O) :
    readout (factoredEdit u v) φ x o = ∑ r, u o r * (∑ f, v r f * φ x f) := by
  simp only [readout, factoredEdit, Finset.sum_mul, Finset.mul_sum]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro r _
  apply Finset.sum_congr rfl
  intro f _
  ring

/-- An exact right-factor kernel condition suffices for arbitrary left factors. -/
theorem factored_preservation [Fintype R] (w : O → F → ℝ)
    (u : O → R → ℝ) (v : R → F → ℝ) (φ : X → F → ℝ) (x : X)
    (hkernel : ∀ r, ∑ f, v r f * φ x f = 0) (o : O) :
    readout (edit w (factoredEdit u v)) φ x o = readout w φ x o := by
  apply sub_eq_zero.mp
  rw [edit_exact, factored_effect]
  simp only [hkernel, mul_zero, Finset.sum_const_zero]

theorem factored_span_preservation [Fintype A] [Fintype R] (w : O → F → ℝ)
    (u : O → R → ℝ) (v : R → F → ℝ) (φ : X → F → ℝ)
    (anchor : A → X) (c : A → ℝ) (x : X)
    (hspan : InFeatureSpan φ anchor c x)
    (hkernel : ∀ a r, ∑ f, v r f * φ (anchor a) f = 0) (o : O) :
    readout (edit w (factoredEdit u v)) φ x o = readout w φ x o := by
  apply protected_span_edit w (factoredEdit u v) φ anchor c x o hspan
  intro a
  rw [factored_effect]
  simp only [hkernel, mul_zero, Finset.sum_const_zero]

/-- Approximate right-factor annihilation yields a left-factor-weighted budget. -/
theorem factored_residual_bound [Fintype R] (u : O → R → ℝ) (v : R → F → ℝ)
    (φ : X → F → ℝ) (x : X) (η : R → ℝ)
    (hkernel : ∀ r, |∑ f, v r f * φ x f| ≤ η r) (o : O) :
    |readout (factoredEdit u v) φ x o| ≤ ∑ r, |u o r| * η r := by
  rw [factored_effect]
  apply (Finset.abs_sum_le_sum_abs _ _).trans
  apply Finset.sum_le_sum
  intro r _
  rw [abs_mul]
  exact mul_le_mul_of_nonneg_left (hkernel r) (abs_nonneg _)

end WeightSpace.Generic
