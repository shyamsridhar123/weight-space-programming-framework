import WeightSpace.Contracts

/-! # Concrete rational certificates

These examples are checked by Lean using the general theorems. They certify
explicit real networks with rational coefficients, not a sampled approximation.
-/

noncomputable section

namespace WeightSpace.Examples

def base : Params (Fin 1) (Fin 1) where
  w1 _ _ := 1
  b1 _ := 0
  w2 _ := 4 / 5
  b2 := 0

def changed : Params (Fin 1) (Fin 1) where
  w1 _ _ := 3 / 2
  b1 _ := 1 / 4
  w2 _ := 9 / 5
  b2 := 1 / 10

def unitBox : InputBox (Fin 1) where
  lower _ := 0
  upper _ := 1

theorem unitBox_valid : unitBox.Valid := by
  intro f
  norm_num [unitBox]

/-- A nonzero edit of all four parameter blocks has exact budget 49/20. -/
theorem all_layer_budget : driftBudget base changed unitBox = 49 / 20 := by
  norm_num [driftBudget, hiddenUpper, hiddenDrift, inputRadius, base, changed, unitBox,
    Fin.sum_univ_succ]

theorem all_layer_certificate (x : Fin 1 → ℝ) (hx : InBox unitBox x) :
    |score changed x - score base x| ≤ 49 / 20 := by
  apply score_drift_le_of_budget base changed unitBox (49 / 20) _ x hx
  rw [all_layer_budget]

def readoutEdit : ReadoutEdit (Fin 1) where
  dw2 _ := 1
  db2 := 0

theorem readout_budget : readoutBudget base readoutEdit unitBox = 1 := by
  norm_num [readoutBudget, hiddenUpper, base, readoutEdit, unitBox, Fin.sum_univ_succ]

theorem readout_certificate (x : Fin 1 → ℝ) (hx : InBox unitBox x) :
    |score (editReadout base readoutEdit) x - score base x| ≤ 1 := by
  simpa only [readout_budget] using readout_drift_le base readoutEdit unitBox x hx

def zeroBox : InputBox (Fin 1) where
  lower _ := 0
  upper _ := 0

theorem protected_region (x : Fin 1 → ℝ) (hx : InBox zeroBox x) :
    score (editReadout base readoutEdit) x = score base x := by
  have hb : readoutBudget base readoutEdit zeroBox = 0 := by
    norm_num [readoutBudget, hiddenUpper, base, readoutEdit, zeroBox, Fin.sum_univ_succ]
  have hd := readout_drift_le base readoutEdit zeroBox x hx
  rw [hb] at hd
  exact sub_eq_zero.mp (abs_nonpos_iff.mp hd)

end WeightSpace.Examples
