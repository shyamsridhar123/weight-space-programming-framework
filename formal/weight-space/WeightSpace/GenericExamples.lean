import WeightSpace.Generic
import WeightSpace.Sequential

/-!
# Non-RTS rational witnesses

These are explicit real-valued examples checked by the kernel. They illustrate
an arbitrary nonlinear feature map, exact and approximate protection, a factored
edit, and a three-layer all-parameter perturbation with mixed activations.
They are not generated from, or a proof of, any Python execution trace.
-/

noncomputable section

open scoped BigOperators

namespace WeightSpace.GenericExamples

open Generic

def features (x : ℝ × ℝ) (f : Fin 3) : ℝ :=
  if f = 0 then 1 else if f = 1 then x.1 else x.2 * x.2

def anchor (a : Fin 2) : ℝ × ℝ := if a = 0 then (0, 0) else (1, 0)

def coefficients (t : ℝ) (a : Fin 2) : ℝ := if a = 0 then 1 - t else t

def baseReadout (o : Fin 2) (f : Fin 3) : ℝ :=
  if o = 0 then (if f = 0 then 1 else if f = 1 then 2 else 3)
  else (if f = 0 then -1 else if f = 1 then 0 else 1)

def selectiveEdit (o : Fin 2) (f : Fin 3) : ℝ :=
  if f = 2 then (if o = 0 then 1 / 2 else -1 / 4) else 0

/-- Every input (t,0) has an explicit two-anchor feature representation. -/
theorem line_in_feature_span (t : ℝ) :
    InFeatureSpan features anchor (coefficients t) (t, 0) := by
  intro f
  by_cases h₀ : f = 0
  · simp [features, coefficients, h₀, Fin.sum_univ_succ]
  · by_cases h₁ : f = 1
    · simp [features, anchor, coefficients, h₁, Fin.sum_univ_succ]
    · simp [features, anchor, coefficients, h₀, h₁, Fin.sum_univ_succ]

theorem selective_anchor_residuals (a : Fin 2) (o : Fin 2) :
    readout selectiveEdit features (anchor a) o = 0 := by
  by_cases ha : a = 0 <;>
    simp [readout, selectiveEdit, features, anchor, ha, Fin.sum_univ_succ]

/-- Exact protection is infinite and includes inputs never enumerated as anchors. -/
theorem nonlinear_feature_protected_line (t : ℝ) (o : Fin 2) :
    readout (edit baseReadout selectiveEdit) features (t, 0) o =
      readout baseReadout features (t, 0) o :=
  protected_span_edit baseReadout selectiveEdit features anchor (coefficients t) (t, 0) o
    (line_in_feature_span t) (fun a => selective_anchor_residuals a o)

theorem target_output_zero_change :
    readout (edit baseReadout selectiveEdit) features (2, 2) 0 -
      readout baseReadout features (2, 2) 0 = 2 := by
  rw [edit_exact]
  norm_num [readout, selectiveEdit, features, Fin.sum_univ_succ,
    (by decide : (0 : Fin 3) ≠ 2), (by decide : (1 : Fin 3) ≠ 2),
    (by decide : (2 : Fin 3) ≠ 1)]

theorem target_output_one_change :
    readout (edit baseReadout selectiveEdit) features (2, 2) 1 -
      readout baseReadout features (2, 2) 1 = -1 := by
  rw [edit_exact]
  norm_num [readout, selectiveEdit, features, Fin.sum_univ_succ,
    (by decide : (0 : Fin 3) ≠ 2), (by decide : (1 : Fin 3) ≠ 2),
    (by decide : (2 : Fin 3) ≠ 1)]

def leakyEdit (_ : Fin 1) (f : Fin 3) : ℝ :=
  if f = 0 then -1 / 100 else if f = 1 then 1 / 50 else 1 / 2

theorem leaky_anchor_bounds (a : Fin 2) :
    |readout leakyEdit features (anchor a) 0| ≤ 1 / 100 := by
  by_cases ha : a = 0 <;>
    norm_num [readout, leakyEdit, features, anchor, ha, Fin.sum_univ_succ,
      (by decide : (2 : Fin 3) ≠ 1)]

theorem extrapolation_coefficient_l1 :
    (∑ a : Fin 2, |coefficients 3 a|) = 5 := by
  norm_num [coefficients, Fin.sum_univ_succ]

/-- Small anchor residuals can grow fivefold through extrapolation. -/
theorem extrapolation_l1_certificate :
    |readout leakyEdit features (3, 0) 0| ≤ 1 / 20 := by
  have h := residual_span_l1_bound leakyEdit features anchor (coefficients 3) (1 / 100)
    (3, 0) 0 (line_in_feature_span 3) leaky_anchor_bounds
  rw [extrapolation_coefficient_l1] at h
  norm_num at h ⊢
  exact h

/-- This witness saturates the coefficient-conditioned bound exactly. -/
theorem extrapolation_l1_bound_is_attained :
    |readout leakyEdit features (3, 0) 0| = 1 / 20 := by
  norm_num [readout, leakyEdit, features, Fin.sum_univ_succ,
    (by decide : (2 : Fin 3) ≠ 1)]

def leftFactor (o : Fin 2) (_ : Fin 1) : ℝ := if o = 0 then 1 / 2 else -1 / 4

def rightFactor (_ : Fin 1) (f : Fin 3) : ℝ := if f = 2 then 1 else 0

theorem factored_edit_eq : factoredEdit leftFactor rightFactor = selectiveEdit := by
  funext o f
  by_cases hf : f = 2 <;>
    simp [factoredEdit, leftFactor, rightFactor, selectiveEdit, hf]

theorem right_factor_protected_kernel (t : ℝ) (r : Fin 1) :
    (∑ f : Fin 3, rightFactor r f * features (t, 0) f) = 0 := by
  norm_num [rightFactor, features, Fin.sum_univ_succ,
    (by decide : (0 : Fin 3) ≠ 2), (by decide : (1 : Fin 3) ≠ 2),
    (by decide : (2 : Fin 3) ≠ 1)]

theorem factored_nonlinear_protection (t : ℝ) (o : Fin 2) :
    readout (edit baseReadout (factoredEdit leftFactor rightFactor)) features (t, 0) o =
      readout baseReadout features (t, 0) o :=
  factored_preservation baseReadout leftFactor rightFactor features (t, 0)
    (right_factor_protected_kernel t) o

/-- An exact-protection contract correctly rules out this nonzero target request. -/
theorem protected_target_one_infeasible (q : Fin 2 → Fin 3 → ℝ)
    (hq : ∀ a, readout q features (anchor a) 0 = readout baseReadout features (anchor a) 0) :
    readout q features (3, 0) 0 - readout baseReadout features (3, 0) 0 ≠ 1 := by
  apply nonzero_target_in_protected_span_impossible baseReadout q features anchor
    (coefficients 3) (3, 0) 0 1 (by norm_num) (line_in_feature_span 3) hq

open Sequential

def scalarBase (n : ℕ) : AffineLayer (Fin 1) (Fin 1) where
  weight _ _ := if n = 0 then 1 else if n = 1 then 2 else -1
  bias _ := if n = 1 then 1 / 2 else 0

def scalarChanged (n : ℕ) : AffineLayer (Fin 1) (Fin 1) where
  weight _ _ := if n = 0 then 5 / 4 else if n = 1 then 7 / 4 else -3 / 2
  bias _ := if n = 0 then 1 / 8 else if n = 1 then 1 / 4 else 1 / 10

def scalarActivation (n : ℕ) (_ : Fin 1) (a : ℝ) : ℝ :=
  if n = 0 then max 0 a else if n = 1 then |a| else a

theorem scalar_activation_magnitude (n : ℕ) (g : Fin 1) :
    MagnitudeBound (scalarActivation n g) := by
  intro a
  by_cases h₀ : n = 0
  · simpa only [scalarActivation, if_pos h₀] using relu_magnitude a
  · by_cases h₁ : n = 1
    · simpa only [scalarActivation, if_neg h₀, if_pos h₁] using absolute_magnitude a
    · simpa only [scalarActivation, if_neg h₀, if_neg h₁] using identity_magnitude a

theorem scalar_activation_lipschitz (n : ℕ) (g : Fin 1) :
    OneLipschitz (scalarActivation n g) := by
  intro a b
  by_cases h₀ : n = 0
  · simpa only [scalarActivation, if_pos h₀] using Sequential.relu_lipschitz a b
  · by_cases h₁ : n = 1
    · simpa only [scalarActivation, if_neg h₀, if_pos h₁] using absolute_lipschitz a b
    · simpa only [scalarActivation, if_neg h₀, if_neg h₁] using identity_lipschitz a b

theorem three_layer_baseline_radius :
    radiusChain (D := fun _ => Fin 1) scalarBase (fun _ => 1) 3 0 = 5 / 2 := by
  norm_num [radiusChain, radiusStep, scalarBase, Fin.sum_univ_succ]

/-- All six affine parameter blocks change; every numeric premise is computed. -/
theorem three_layer_drift_budget :
    driftChain (D := fun _ => Fin 1) scalarBase scalarChanged (fun _ => 1) (fun _ => 0) 3 0 =
      987 / 320 := by
  norm_num [driftChain, driftStep, radiusChain, radiusStep, scalarBase, scalarChanged,
    Fin.sum_univ_succ]

theorem three_layer_uniform_certificate (x : Fin 1 → ℝ) (hx : ∀ f, |x f| ≤ 1) :
    |evalChain (D := fun _ => Fin 1) scalarChanged scalarActivation x 3 0 -
      evalChain (D := fun _ => Fin 1) scalarBase scalarActivation x 3 0| ≤ 987 / 320 := by
  have h := chain_same_input_drift (D := fun _ => Fin 1) scalarBase scalarChanged
    scalarActivation x (fun _ => 1) scalar_activation_magnitude scalar_activation_lipschitz hx 3 0
  simpa only [three_layer_drift_budget] using h

theorem three_layer_endpoint_difference :
    evalChain (D := fun _ => Fin 1) scalarChanged scalarActivation (fun _ => 1) 3 0 -
      evalChain (D := fun _ => Fin 1) scalarBase scalarActivation (fun _ => 1) 3 0 = -443 / 320 := by
  norm_num [evalChain, layerEval, affineValue, scalarBase, scalarChanged, scalarActivation,
    Fin.sum_univ_succ]

end WeightSpace.GenericExamples
