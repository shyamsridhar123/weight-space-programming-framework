import Mathlib.Data.Real.Basic
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Tactic.Ring

/-!
# A concrete finite two-layer ReLU scorer

Every parameter below is a real number. For a multi-output network, instantiate
`Params` once per output coordinate, sharing `w1` and `b1`. No statement in this
file identifies arbitrary floating-point execution with real arithmetic.
-/

noncomputable section

open scoped BigOperators

namespace WeightSpace

/-- A scalar output of a finite two-layer network, including both biases. -/
structure Params (F H : Type*) where
  w1 : H → F → ℝ
  b1 : H → ℝ
  w2 : H → ℝ
  b2 : ℝ

variable {F H : Type*} [Fintype F] [Fintype H]

/-- Hidden preactivation. The feature order and its meaning are fixed. -/
def preactivation (p : Params F H) (x : F → ℝ) (h : H) : ℝ :=
  p.b1 h + ∑ f, p.w1 h f * x f

/-- Hidden ReLU activation. -/
def hidden (p : Params F H) (x : F → ℝ) (h : H) : ℝ :=
  max 0 (preactivation p x h)

/-- The same scalar formula used by the exact rational certificate checker. -/
def score (p : Params F H) (x : F → ℝ) : ℝ :=
  p.b2 + ∑ h, p.w2 h * hidden p x h

/-- Relabel the hidden units, transporting every incident parameter. -/
def permute (p : Params F H) (π : H ≃ H) : Params F H where
  w1 h f := p.w1 (π h) f
  b1 h := p.b1 (π h)
  w2 h := p.w2 (π h)
  b2 := p.b2

@[simp]
theorem hidden_permute (p : Params F H) (π : H ≃ H) (x : F → ℝ) (h : H) :
    hidden (permute p π) x h = hidden p x (π h) := rfl

/-- Hidden-unit permutation is an exact semantic symmetry of this scorer. -/
theorem score_permute (p : Params F H) (π : H ≃ H) (x : F → ℝ) :
    score (permute p π) x = score p x := by
  change p.b2 + ∑ h, p.w2 (π h) * hidden p x (π h) =
    p.b2 + ∑ h, p.w2 h * hidden p x h
  rw [Equiv.sum_comp π (fun h => p.w2 h * hidden p x h)]

/-- All-layer additive editing; an edit has the same shape as its parameters. -/
def applyEdit (p d : Params F H) : Params F H where
  w1 h f := p.w1 h f + d.w1 h f
  b1 h := p.b1 h + d.b1 h
  w2 h := p.w2 h + d.w2 h
  b2 := p.b2 + d.b2

/-- Coordinate edits must be transported together with their base network. -/
theorem permute_applyEdit (p d : Params F H) (π : H ≃ H) :
    permute (applyEdit p d) π = applyEdit (permute p π) (permute d π) := rfl

/-- Editing in a transported gauge has exactly the same function semantics. -/
theorem transported_edit_score (p d : Params F H) (π : H ≃ H) (x : F → ℝ) :
    score (applyEdit (permute p π) (permute d π)) x = score (applyEdit p d) x := by
  rw [← permute_applyEdit]
  exact score_permute (applyEdit p d) π x

/-- An edit whose hidden representation is frozen. -/
structure ReadoutEdit (H : Type*) where
  dw2 : H → ℝ
  db2 : ℝ

def editReadout (p : Params F H) (d : ReadoutEdit H) : Params F H where
  w1 := p.w1
  b1 := p.b1
  w2 h := p.w2 h + d.dw2 h
  b2 := p.b2 + d.db2

def readoutEffect (p : Params F H) (d : ReadoutEdit H) (x : F → ℝ) : ℝ :=
  d.db2 + ∑ h, d.dw2 h * hidden p x h

/-- Frozen-hidden editing has an exact linear effect, without approximation. -/
theorem readout_exact (p : Params F H) (d : ReadoutEdit H) (x : F → ℝ) :
    score (editReadout p d) x - score p x = readoutEffect p d x := by
  change (p.b2 + d.db2 + ∑ h, (p.w2 h + d.dw2 h) * hidden p x h) -
    (p.b2 + ∑ h, p.w2 h * hidden p x h) = d.db2 + ∑ h, d.dw2 h * hidden p x h
  simp only [add_mul, Finset.sum_add_distrib]
  ring

def addReadoutEdits (d e : ReadoutEdit H) : ReadoutEdit H where
  dw2 h := d.dw2 h + e.dw2 h
  db2 := d.db2 + e.db2

/-- The linear effects of readout edits add in the same hidden representation. -/
theorem readoutEffect_add (p : Params F H) (d e : ReadoutEdit H) (x : F → ℝ) :
    readoutEffect p (addReadoutEdits d e) x =
      readoutEffect p d x + readoutEffect p e x := by
  simp only [readoutEffect, addReadoutEdits, add_mul, Finset.sum_add_distrib]
  ring

/-- Sequential readout edits agree with adding the edits before application. -/
theorem editReadout_add (p : Params F H) (d e : ReadoutEdit H) :
    editReadout (editReadout p d) e = editReadout p (addReadoutEdits d e) := by
  cases p
  cases d
  cases e
  simp only [editReadout, addReadoutEdits, add_assoc]

end WeightSpace
