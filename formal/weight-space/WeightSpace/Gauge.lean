import WeightSpace.Core

/-! # Positive hidden-unit rescaling and transported edits -/

noncomputable section

open scoped BigOperators

namespace WeightSpace

variable {F H : Type*} [Fintype F] [Fintype H]

/-- Scale a hidden unit's incoming weights and bias, inversely scaling its readout. -/
def rescale (p : Params F H) (s : H → ℝ) : Params F H where
  w1 h f := s h * p.w1 h f
  b1 h := s h * p.b1 h
  w2 h := p.w2 h / s h
  b2 := p.b2

theorem preactivation_rescale (p : Params F H) (s : H → ℝ) (x : F → ℝ) (h : H) :
    preactivation (rescale p s) x h = s h * preactivation p x h := by
  simp only [preactivation, rescale, mul_add, Finset.mul_sum, mul_assoc]

theorem hidden_rescale (p : Params F H) (s : H → ℝ) (hs : ∀ h, 0 ≤ s h)
    (x : F → ℝ) (h : H) : hidden (rescale p s) x h = s h * hidden p x h := by
  unfold hidden
  rw [preactivation_rescale, mul_max_of_nonneg _ _ (hs h), mul_zero]

/-- Strictly positive rescaling is an exact semantic symmetry over the reals. -/
theorem score_rescale (p : Params F H) (s : H → ℝ) (hs : ∀ h, 0 < s h)
    (x : F → ℝ) : score (rescale p s) x = score p x := by
  change p.b2 + ∑ h, p.w2 h / s h * hidden (rescale p s) x h =
    p.b2 + ∑ h, p.w2 h * hidden p x h
  congr 1
  apply Finset.sum_congr rfl
  intro h _
  rw [hidden_rescale p s (fun h => (hs h).le), ← mul_assoc,
    div_mul_cancel₀ _ (ne_of_gt (hs h))]

/-- Additive parameter edits transform by the same linear coordinate map. -/
theorem rescale_applyEdit (p d : Params F H) (s : H → ℝ) :
    rescale (applyEdit p d) s = applyEdit (rescale p s) (rescale d s) := by
  cases p
  cases d
  simp only [rescale, applyEdit, mul_add, add_div]

/-- Joint positive rescaling of a network and its edit preserves edited semantics. -/
theorem rescaled_edit_score (p d : Params F H) (s : H → ℝ) (hs : ∀ h, 0 < s h)
    (x : F → ℝ) :
    score (applyEdit (rescale p s) (rescale d s)) x = score (applyEdit p d) x := by
  rw [← rescale_applyEdit]
  exact score_rescale (applyEdit p d) s hs x

end WeightSpace
