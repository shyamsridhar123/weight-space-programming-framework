import WeightSpace.Bounds

/-! # Composable semantic contracts and strict argmax preservation -/

noncomputable section

open scoped BigOperators

namespace WeightSpace

/-- Uniform scalar-output drift over an explicitly chosen domain. -/
def DriftContract {X : Type*} (f g : X → ℝ) (domain : X → Prop) (ε : ℝ) : Prop :=
  ∀ x, domain x → |g x - f x| ≤ ε

/-- Sequential drift budgets compose by the triangle inequality. -/
theorem DriftContract.trans {X : Type*} {f g h : X → ℝ} {domain : X → Prop}
    {ε₁ ε₂ : ℝ} (hfg : DriftContract f g domain ε₁)
    (hgh : DriftContract g h domain ε₂) : DriftContract f h domain (ε₁ + ε₂) := by
  intro x hx
  have h₁ := hfg x hx
  have h₂ := hgh x hx
  calc
    |h x - f x| = |(g x - f x) + (h x - g x)| := by congr 1; ring
    _ ≤ |g x - f x| + |h x - g x| := abs_add_le _ _
    _ ≤ ε₁ + ε₂ := add_le_add h₁ h₂

/-- A finite sequence of valid edits has the sum of its individual budgets. -/
theorem DriftContract.chain {X : Type*} (state : ℕ → X → ℝ) (domain : X → Prop)
    (budget : ℕ → ℝ) (hstep : ∀ k, DriftContract (state k) (state (k + 1)) domain (budget k))
    (n : ℕ) : DriftContract (state 0) (state n) domain (∑ k ∈ Finset.range n, budget k) := by
  induction n with
  | zero =>
      intro x _
      simp
  | succ n ih =>
      simpa only [Finset.sum_range_succ] using ih.trans (hstep n)

/-- A strict winner survives unequal per-coordinate output perturbation bounds. -/
theorem argmax_preserved {C : Type*} (base changed : C → ℝ) (ε : C → ℝ)
    (winner : C) (hdrift : ∀ c, |changed c - base c| ≤ ε c)
    (hmargin : ∀ c, c ≠ winner → ε winner + ε c < base winner - base c) :
    ∀ c, c ≠ winner → changed c < changed winner := by
  intro c hc
  have hw := (abs_le.mp (hdrift winner)).1
  have ho := (abs_le.mp (hdrift c)).2
  have hm := hmargin c hc
  linarith

/-- Familiar `margin > 2 ε` specialization of the coordinatewise theorem. -/
theorem argmax_preserved_uniform {C : Type*} (base changed : C → ℝ) (ε : ℝ)
    (winner : C) (hdrift : ∀ c, |changed c - base c| ≤ ε)
    (hmargin : ∀ c, c ≠ winner → 2 * ε < base winner - base c) :
    ∀ c, c ≠ winner → changed c < changed winner := by
  apply argmax_preserved base changed (fun _ => ε) winner hdrift
  intro c hc
  have hm := hmargin c hc
  linarith

end WeightSpace
