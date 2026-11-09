import WeightSpace.Bounds

/-!
# Arbitrary-depth affine / one-Lipschitz network drift

Each depth may have its own finite coordinate type and activation. The baseline
and edited networks share those types and activations. Width, depth, and input
meaning are not tied to an application. This is a conservative componentwise
real-arithmetic certificate, not a floating-point execution theorem.
-/

noncomputable section

open scoped BigOperators

namespace WeightSpace.Sequential

structure AffineLayer (F G : Type*) where
  weight : G → F → ℝ
  bias : G → ℝ

variable {F G : Type*} [Fintype F]

def affineValue (p : AffineLayer F G) (x : F → ℝ) (g : G) : ℝ :=
  p.bias g + ∑ f, p.weight g f * x f

def layerEval (σ : G → ℝ → ℝ) (p : AffineLayer F G) (x : F → ℝ) (g : G) : ℝ :=
  σ g (affineValue p x g)

def radiusStep (p : AffineLayer F G) (r : F → ℝ) (g : G) : ℝ :=
  |p.bias g| + ∑ f, |p.weight g f| * r f

/-- Edited weights multiply propagated drift; baseline radii multiply weight drift. -/
def driftStep (p q : AffineLayer F G) (r d : F → ℝ) (g : G) : ℝ :=
  |q.bias g - p.bias g| +
    ∑ f, (|q.weight g f - p.weight g f| * r f + |q.weight g f| * d f)

def MagnitudeBound (σ : ℝ → ℝ) : Prop := ∀ a, |σ a| ≤ |a|

def OneLipschitz (σ : ℝ → ℝ) : Prop := ∀ a b, |σ a - σ b| ≤ |a - b|

theorem identity_magnitude : MagnitudeBound (fun a => a) := fun _ => le_rfl

theorem identity_lipschitz : OneLipschitz (fun a => a) := fun _ _ => le_rfl

theorem relu_magnitude : MagnitudeBound (fun a => max 0 a) := by
  intro a
  rw [abs_of_nonneg (le_max_left _ _)]
  exact max_le (abs_nonneg _) (le_abs_self _)

theorem relu_lipschitz : OneLipschitz (fun a => max 0 a) := WeightSpace.relu_lipschitz

theorem absolute_magnitude : MagnitudeBound (fun a => |a|) := by
  intro a
  simp only [abs_abs, le_refl]

theorem absolute_lipschitz : OneLipschitz (fun a => |a|) := by
  intro a b
  exact abs_abs_sub_abs_le_abs_sub a b

theorem radiusStep_nonneg (p : AffineLayer F G) (r : F → ℝ)
    (hr : ∀ f, 0 ≤ r f) (g : G) : 0 ≤ radiusStep p r g := by
  apply add_nonneg (abs_nonneg _)
  exact Finset.sum_nonneg (fun f _ => mul_nonneg (abs_nonneg _) (hr f))

theorem driftStep_nonneg (p q : AffineLayer F G) (r d : F → ℝ)
    (hr : ∀ f, 0 ≤ r f) (hd : ∀ f, 0 ≤ d f) (g : G) : 0 ≤ driftStep p q r d g := by
  apply add_nonneg (abs_nonneg _)
  apply Finset.sum_nonneg
  intro f _
  exact add_nonneg (mul_nonneg (abs_nonneg _) (hr f)) (mul_nonneg (abs_nonneg _) (hd f))

theorem layer_radius_le (σ : G → ℝ → ℝ) (p : AffineLayer F G)
    (x r : F → ℝ) (hσ : ∀ g, MagnitudeBound (σ g)) (hx : ∀ f, |x f| ≤ r f)
    (g : G) : |layerEval σ p x g| ≤ radiusStep p r g := by
  calc
    |layerEval σ p x g| ≤ |affineValue p x g| := hσ g _
    _ ≤ |p.bias g| + |∑ f, p.weight g f * x f| := abs_add_le _ _
    _ ≤ |p.bias g| + ∑ f, |p.weight g f * x f| :=
      add_le_add_left (Finset.abs_sum_le_sum_abs _ _) _
    _ ≤ radiusStep p r g := by
      apply add_le_add_left
      apply Finset.sum_le_sum
      intro f _
      rw [abs_mul]
      exact mul_le_mul_of_nonneg_left (hx f) (abs_nonneg _)

theorem affine_term_drift_le (a b x y r d : ℝ)
    (hx : |x| ≤ r) (hd : |y - x| ≤ d) :
    |b * y - a * x| ≤ |b - a| * r + |b| * d := by
  have hid : b * y - a * x = (b - a) * x + b * (y - x) := by ring
  rw [hid]
  calc
    |(b - a) * x + b * (y - x)| ≤ |(b - a) * x| + |b * (y - x)| := abs_add_le _ _
    _ = |b - a| * |x| + |b| * |y - x| := by rw [abs_mul, abs_mul]
    _ ≤ |b - a| * r + |b| * d :=
      add_le_add (mul_le_mul_of_nonneg_left hx (abs_nonneg _))
        (mul_le_mul_of_nonneg_left hd (abs_nonneg _))

/-- Joint input and parameter drift through one shared one-Lipschitz activation. -/
theorem layer_drift_le (σ : G → ℝ → ℝ) (p q : AffineLayer F G)
    (x y r d : F → ℝ) (hσ : ∀ g, OneLipschitz (σ g))
    (hx : ∀ f, |x f| ≤ r f) (hd : ∀ f, |y f - x f| ≤ d f) (g : G) :
    |layerEval σ q y g - layerEval σ p x g| ≤ driftStep p q r d g := by
  have hid : affineValue q y g - affineValue p x g = (q.bias g - p.bias g) +
      ∑ f, (q.weight g f * y f - p.weight g f * x f) := by
    simp only [affineValue, Finset.sum_sub_distrib]
    ring
  calc
    |layerEval σ q y g - layerEval σ p x g|
        ≤ |affineValue q y g - affineValue p x g| := hσ g _ _
    _ = |(q.bias g - p.bias g) + ∑ f, (q.weight g f * y f - p.weight g f * x f)| := by
      rw [hid]
    _ ≤ |q.bias g - p.bias g| + |∑ f, (q.weight g f * y f - p.weight g f * x f)| :=
      abs_add_le _ _
    _ ≤ |q.bias g - p.bias g| + ∑ f, |q.weight g f * y f - p.weight g f * x f| :=
      add_le_add_left (Finset.abs_sum_le_sum_abs _ _) _
    _ ≤ driftStep p q r d g := by
      apply add_le_add_left
      exact Finset.sum_le_sum (fun f _ =>
        affine_term_drift_le _ _ _ _ _ _ (hx f) (hd f))

variable {D : ℕ → Type*} [∀ n, Fintype (D n)]

/-- A prefix ending at depth n; each layer may change the coordinate dimension. -/
def evalChain (p : (n : ℕ) → AffineLayer (D n) (D (n + 1)))
    (σ : (n : ℕ) → D (n + 1) → ℝ → ℝ) (x : D 0 → ℝ) : (n : ℕ) → D n → ℝ
  | 0 => x
  | n + 1 => layerEval (σ n) (p n) (evalChain p σ x n)

def radiusChain (p : (n : ℕ) → AffineLayer (D n) (D (n + 1)))
    (r₀ : D 0 → ℝ) : (n : ℕ) → D n → ℝ
  | 0 => r₀
  | n + 1 => radiusStep (p n) (radiusChain p r₀ n)

def driftChain (p q : (n : ℕ) → AffineLayer (D n) (D (n + 1)))
    (r₀ d₀ : D 0 → ℝ) : (n : ℕ) → D n → ℝ
  | 0 => d₀
  | n + 1 => driftStep (p n) (q n) (radiusChain p r₀ n) (driftChain p q r₀ d₀ n)

theorem chain_radius_nonneg (p : (n : ℕ) → AffineLayer (D n) (D (n + 1)))
    (r₀ : D 0 → ℝ) (hr : ∀ f, 0 ≤ r₀ f) (n : ℕ) :
    ∀ f, 0 ≤ radiusChain p r₀ n f := by
  induction n with
  | zero => exact hr
  | succ n ih => exact radiusStep_nonneg (p n) (radiusChain p r₀ n) ih

theorem chain_drift_nonneg (p q : (n : ℕ) → AffineLayer (D n) (D (n + 1)))
    (r₀ d₀ : D 0 → ℝ) (hr : ∀ f, 0 ≤ r₀ f) (hd : ∀ f, 0 ≤ d₀ f) (n : ℕ) :
    ∀ f, 0 ≤ driftChain p q r₀ d₀ n f := by
  induction n with
  | zero => exact hd
  | succ n ih =>
      exact driftStep_nonneg (p n) (q n) (radiusChain p r₀ n) (driftChain p q r₀ d₀ n)
        (chain_radius_nonneg p r₀ hr n) ih

theorem chain_radius_le (p : (n : ℕ) → AffineLayer (D n) (D (n + 1)))
    (σ : (n : ℕ) → D (n + 1) → ℝ → ℝ) (x r₀ : D 0 → ℝ)
    (hσ : ∀ n g, MagnitudeBound (σ n g)) (hx : ∀ f, |x f| ≤ r₀ f) (n : ℕ) :
    ∀ f, |evalChain p σ x n f| ≤ radiusChain p r₀ n f := by
  induction n with
  | zero => exact hx
  | succ n ih =>
      exact layer_radius_le (σ n) (p n) (evalChain p σ x n) (radiusChain p r₀ n) (hσ n) ih

/-- Arbitrary-depth induction, also allowing a separately bounded input perturbation. -/
theorem chain_drift_le (p q : (n : ℕ) → AffineLayer (D n) (D (n + 1)))
    (σ : (n : ℕ) → D (n + 1) → ℝ → ℝ) (x y r₀ d₀ : D 0 → ℝ)
    (hmag : ∀ n g, MagnitudeBound (σ n g)) (hlip : ∀ n g, OneLipschitz (σ n g))
    (hx : ∀ f, |x f| ≤ r₀ f) (hd : ∀ f, |y f - x f| ≤ d₀ f) (n : ℕ) :
    ∀ f, |evalChain q σ y n f - evalChain p σ x n f| ≤ driftChain p q r₀ d₀ n f := by
  induction n with
  | zero => exact hd
  | succ n ih =>
      exact layer_drift_le (σ n) (p n) (q n) (evalChain p σ x n) (evalChain q σ y n)
        (radiusChain p r₀ n) (driftChain p q r₀ d₀ n) (hlip n)
        (chain_radius_le p σ x r₀ hmag hx n) ih

theorem chain_same_input_drift (p q : (n : ℕ) → AffineLayer (D n) (D (n + 1)))
    (σ : (n : ℕ) → D (n + 1) → ℝ → ℝ) (x r₀ : D 0 → ℝ)
    (hmag : ∀ n g, MagnitudeBound (σ n g)) (hlip : ∀ n g, OneLipschitz (σ n g))
    (hx : ∀ f, |x f| ≤ r₀ f) (n : ℕ) :
    ∀ f, |evalChain q σ x n f - evalChain p σ x n f| ≤ driftChain p q r₀ (fun _ => 0) n f := by
  apply chain_drift_le p q σ x x r₀ (fun _ => 0) hmag hlip hx
  intro f
  simp only [sub_self, abs_zero, le_refl]

/-- Output-coordinate budgets also compose across independent edit transactions. -/
theorem coordinate_drift_compose {Y : Type*} (p q s : Y → ℝ) (ε η : Y → ℝ)
    (hpq : ∀ y, |q y - p y| ≤ ε y) (hqs : ∀ y, |s y - q y| ≤ η y) (y : Y) :
    |s y - p y| ≤ ε y + η y := by
  have hid : s y - p y = (q y - p y) + (s y - q y) := by ring
  rw [hid]
  exact (abs_add_le _ _).trans (add_le_add (hpq y) (hqs y))

end WeightSpace.Sequential
