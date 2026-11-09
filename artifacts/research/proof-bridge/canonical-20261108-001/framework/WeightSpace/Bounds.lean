import WeightSpace.Core
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum

/-!
# Sound input-box certificates for all-layer edits

The budget is conservative interval arithmetic over real numbers. It applies
to every point in a box, not just a finite validation sample. Different output
coordinates may use different budgets.
-/

noncomputable section

open scoped BigOperators

namespace WeightSpace

structure InputBox (F : Type*) where
  lower : F → ℝ
  upper : F → ℝ

def InputBox.Valid {F : Type*} (b : InputBox F) : Prop :=
  ∀ f, b.lower f ≤ b.upper f

def InBox {F : Type*} (b : InputBox F) (x : F → ℝ) : Prop :=
  ∀ f, b.lower f ≤ x f ∧ x f ≤ b.upper f

def inputRadius {F : Type*} (b : InputBox F) (f : F) : ℝ :=
  max |b.lower f| |b.upper f|

variable {F H : Type*} [Fintype F] [Fintype H]

def hiddenUpper (p : Params F H) (b : InputBox F) (h : H) : ℝ :=
  max 0 (p.b1 h + ∑ f, max (p.w1 h f * b.lower f) (p.w1 h f * b.upper f))

def hiddenDrift (p q : Params F H) (b : InputBox F) (h : H) : ℝ :=
  |q.b1 h - p.b1 h| + ∑ f, |q.w1 h f - p.w1 h f| * inputRadius b f

/-- `q` is the edited endpoint; its readout weights multiply the hidden drift. -/
def driftBudget (p q : Params F H) (b : InputBox F) : ℝ :=
  |q.b2 - p.b2| + ∑ h,
    (|q.w2 h - p.w2 h| * hiddenUpper p b h + |q.w2 h| * hiddenDrift p q b h)

theorem abs_input_le_radius (b : InputBox F) (x : F → ℝ) (hx : InBox b x) (f : F) :
    |x f| ≤ inputRadius b f := by
  unfold inputRadius
  apply abs_le.mpr
  constructor
  · have hlo := (hx f).1
    have ha := neg_abs_le (b.lower f)
    have hm := le_max_left |b.lower f| |b.upper f|
    linarith
  · exact (hx f).2 |>.trans ((le_abs_self _).trans (le_max_right _ _))

theorem weighted_box_le (w l u x : ℝ) (hx : l ≤ x ∧ x ≤ u) :
    w * x ≤ max (w * l) (w * u) := by
  by_cases hw : 0 ≤ w
  · exact (mul_le_mul_of_nonneg_left hx.2 hw).trans (le_max_right _ _)
  · exact (mul_le_mul_of_nonpos_left hx.1 (le_of_lt (lt_of_not_ge hw))).trans
      (le_max_left _ _)

theorem hidden_nonneg (p : Params F H) (x : F → ℝ) (h : H) :
    0 ≤ hidden p x h := le_max_left _ _

theorem hidden_le_upper (p : Params F H) (b : InputBox F) (x : F → ℝ)
    (hx : InBox b x) (h : H) : hidden p x h ≤ hiddenUpper p b h := by
  unfold hidden preactivation hiddenUpper
  apply max_le_max le_rfl
  apply add_le_add_left
  exact Finset.sum_le_sum (fun f _ => weighted_box_le _ _ _ _ (hx f))

theorem abs_hidden_le_upper (p : Params F H) (b : InputBox F) (x : F → ℝ)
    (hx : InBox b x) (h : H) : |hidden p x h| ≤ hiddenUpper p b h := by
  rw [abs_of_nonneg (hidden_nonneg p x h)]
  exact hidden_le_upper p b x hx h

/-- ReLU is globally one-Lipschitz, including across activation boundaries. -/
theorem relu_lipschitz (a b : ℝ) : |max 0 a - max 0 b| ≤ |a - b| := by
  by_cases ha : 0 ≤ a
  · rw [max_eq_right ha]
    by_cases hb : 0 ≤ b
    · rw [max_eq_right hb]
    · rw [max_eq_left (le_of_lt (lt_of_not_ge hb)), sub_zero, abs_of_nonneg ha]
      have hab : a ≤ a - b := by linarith
      exact hab.trans (le_abs_self _)
  · rw [max_eq_left (le_of_lt (lt_of_not_ge ha))]
    by_cases hb : 0 ≤ b
    · rw [max_eq_right hb, zero_sub, abs_neg, abs_of_nonneg hb]
      have hab : b ≤ -(a - b) := by linarith
      exact hab.trans (neg_le_abs _)
    · rw [max_eq_left (le_of_lt (lt_of_not_ge hb))]
      simp only [sub_self, abs_zero]
      exact abs_nonneg _

theorem preactivation_drift_le (p q : Params F H) (b : InputBox F)
    (x : F → ℝ) (hx : InBox b x) (h : H) :
    |preactivation q x h - preactivation p x h| ≤ hiddenDrift p q b h := by
  have hid : preactivation q x h - preactivation p x h =
      (q.b1 h - p.b1 h) + ∑ f, (q.w1 h f - p.w1 h f) * x f := by
    simp only [preactivation, sub_mul, Finset.sum_sub_distrib]
    ring
  rw [hid]
  calc
    |q.b1 h - p.b1 h + ∑ f, (q.w1 h f - p.w1 h f) * x f|
        ≤ |q.b1 h - p.b1 h| + |∑ f, (q.w1 h f - p.w1 h f) * x f| := abs_add_le _ _
    _ ≤ |q.b1 h - p.b1 h| + ∑ f, |(q.w1 h f - p.w1 h f) * x f| :=
      add_le_add_left (Finset.abs_sum_le_sum_abs _ _) _
    _ ≤ hiddenDrift p q b h := by
      unfold hiddenDrift
      apply add_le_add_left
      apply Finset.sum_le_sum
      intro f _
      rw [abs_mul]
      exact mul_le_mul_of_nonneg_left (abs_input_le_radius b x hx f) (abs_nonneg _)

theorem hidden_drift_le (p q : Params F H) (b : InputBox F)
    (x : F → ℝ) (hx : InBox b x) (h : H) :
    |hidden q x h - hidden p x h| ≤ hiddenDrift p q b h :=
  (relu_lipschitz _ _).trans (preactivation_drift_le p q b x hx h)

theorem output_term_drift_le (p q : Params F H) (b : InputBox F)
    (x : F → ℝ) (hx : InBox b x) (h : H) :
    |q.w2 h * hidden q x h - p.w2 h * hidden p x h| ≤
      |q.w2 h - p.w2 h| * hiddenUpper p b h + |q.w2 h| * hiddenDrift p q b h := by
  have hid : q.w2 h * hidden q x h - p.w2 h * hidden p x h =
      (q.w2 h - p.w2 h) * hidden p x h + q.w2 h * (hidden q x h - hidden p x h) := by
    ring
  rw [hid]
  calc
    |(q.w2 h - p.w2 h) * hidden p x h + q.w2 h * (hidden q x h - hidden p x h)|
        ≤ |(q.w2 h - p.w2 h) * hidden p x h| +
          |q.w2 h * (hidden q x h - hidden p x h)| := abs_add_le _ _
    _ = |q.w2 h - p.w2 h| * |hidden p x h| +
        |q.w2 h| * |hidden q x h - hidden p x h| := by rw [abs_mul, abs_mul]
    _ ≤ |q.w2 h - p.w2 h| * hiddenUpper p b h +
        |q.w2 h| * hiddenDrift p q b h :=
      add_le_add
        (mul_le_mul_of_nonneg_left (abs_hidden_le_upper p b x hx h) (abs_nonneg _))
        (mul_le_mul_of_nonneg_left (hidden_drift_le p q b x hx h) (abs_nonneg _))

/-- The central all-layer certificate theorem, valid for every input in the box. -/
theorem score_drift_le (p q : Params F H) (b : InputBox F)
    (x : F → ℝ) (hx : InBox b x) :
    |score q x - score p x| ≤ driftBudget p q b := by
  have hid : score q x - score p x = (q.b2 - p.b2) +
      ∑ h, (q.w2 h * hidden q x h - p.w2 h * hidden p x h) := by
    simp only [score, Finset.sum_sub_distrib]
    ring
  rw [hid]
  calc
    |q.b2 - p.b2 + ∑ h, (q.w2 h * hidden q x h - p.w2 h * hidden p x h)|
        ≤ |q.b2 - p.b2| + |∑ h, (q.w2 h * hidden q x h - p.w2 h * hidden p x h)| :=
      abs_add_le _ _
    _ ≤ |q.b2 - p.b2| + ∑ h, |q.w2 h * hidden q x h - p.w2 h * hidden p x h| :=
      add_le_add_left (Finset.abs_sum_le_sum_abs _ _) _
    _ ≤ driftBudget p q b := by
      unfold driftBudget
      apply add_le_add_left
      exact Finset.sum_le_sum (fun h _ => output_term_drift_le p q b x hx h)

/-- A concrete numeric check of the budget gives the advertised uniform bound. -/
theorem score_drift_le_of_budget (p q : Params F H) (b : InputBox F)
    (ε : ℝ) (hb : driftBudget p q b ≤ ε) (x : F → ℝ) (hx : InBox b x) :
    |score q x - score p x| ≤ ε :=
  (score_drift_le p q b x hx).trans hb

def readoutBudget (p : Params F H) (d : ReadoutEdit H) (b : InputBox F) : ℝ :=
  |d.db2| + ∑ h, |d.dw2 h| * hiddenUpper p b h

/-- The all-layer formula reduces exactly when the hidden layer is unchanged. -/
theorem driftBudget_readout (p : Params F H) (d : ReadoutEdit H) (b : InputBox F) :
    driftBudget p (editReadout p d) b = readoutBudget p d b := by
  simp [driftBudget, readoutBudget, hiddenDrift, editReadout]

theorem readout_drift_le (p : Params F H) (d : ReadoutEdit H) (b : InputBox F)
    (x : F → ℝ) (hx : InBox b x) :
    |score (editReadout p d) x - score p x| ≤ readoutBudget p d b := by
  simpa only [driftBudget_readout] using score_drift_le p (editReadout p d) b x hx

/-- A single readout-coordinate program, useful for sparse real-model certificates. -/
def editReadoutAt [DecidableEq H] (p : Params F H) (h₀ : H) (δ : ℝ) : Params F H :=
  editReadout p { dw2 := fun h => if h = h₀ then δ else 0, db2 := 0 }

theorem readout_single_exact [DecidableEq H] (p : Params F H) (h₀ : H)
    (δ : ℝ) (x : F → ℝ) :
    score (editReadoutAt p h₀ δ) x - score p x = δ * hidden p x h₀ := by
  unfold editReadoutAt
  rw [readout_exact]
  simp [readoutEffect, ite_mul]

/-- Only one hidden row needs numeric evaluation for a sparse readout certificate. -/
theorem readout_single_drift_le [DecidableEq H] (p : Params F H) (h₀ : H)
    (δ : ℝ) (b : InputBox F) (x : F → ℝ) (hx : InBox b x) :
    |score (editReadoutAt p h₀ δ) x - score p x| ≤ |δ| * hiddenUpper p b h₀ := by
  rw [readout_single_exact, abs_mul]
  exact mul_le_mul_of_nonneg_left (abs_hidden_le_upper p b x hx h₀) (abs_nonneg _)

theorem readout_single_drift_le_of_budget [DecidableEq H] (p : Params F H)
    (h₀ : H) (δ : ℝ) (b : InputBox F) (ε : ℝ)
    (hb : |δ| * hiddenUpper p b h₀ ≤ ε) (x : F → ℝ) (hx : InBox b x) :
    |score (editReadoutAt p h₀ δ) x - score p x| ≤ ε :=
  (readout_single_drift_le p h₀ δ b x hx).trans hb

end WeightSpace
