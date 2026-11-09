import WeightSpace.Sequential

/-!
Runtime instance interpretation. Lists use row-major weight order and explicit
post-dot bias vectors. Shape premises are checked in each generated instance;
the Python decoder separately binds these values to immutable runtime endpoints.
Everything here is real arithmetic, without an extractor or native-kernel claim.
-/

noncomputable section
open scoped BigOperators

namespace WeightSpace.Bridge
open Sequential

inductive Activation where
  | identity | relu | absolute
  deriving DecidableEq

def activationValue : Activation → ℝ → ℝ
  | .identity => fun a => a
  | .relu => fun a => max 0 a
  | .absolute => fun a => |a|

theorem activation_magnitude (a : Activation) : MagnitudeBound (activationValue a) := by
  cases a with
  | identity => exact identity_magnitude
  | relu => exact relu_magnitude
  | absolute => exact absolute_magnitude

theorem activation_lipschitz (a : Activation) : OneLipschitz (activationValue a) := by
  cases a with
  | identity => exact identity_lipschitz
  | relu => exact Sequential.relu_lipschitz
  | absolute => exact absolute_lipschitz

-- Direct recursion keeps dependent coordinate types reducible at concrete widths.
def widthAt : List ℕ → ℕ → ℕ
  | [], _ => 1
  | w :: _, 0 => w
  | _ :: ws, n + 1 => widthAt ws n
abbrev Coord (widths : List ℕ) (n : ℕ) := Fin (widthAt widths n)

def layers (widths : List ℕ) (weights biases : List (List ℝ)) (n : ℕ) :
    AffineLayer (Coord widths n) (Coord widths (n + 1)) where
  weight o i := (weights.getD n []).getD (o.val * widthAt widths n + i.val) 0
  bias o := (biases.getD n []).getD o.val 0

def activationsAt (widths : List ℕ) (activations : List Activation) (n : ℕ)
    (_ : Coord widths (n + 1)) : ℝ → ℝ := activationValue (activations.getD n .identity)

def inputBox (widths : List ℕ) (lower upper : List ℝ) : InputBox (Coord widths 0) where
  lower i := lower.getD i.val 0
  upper i := upper.getD i.val 0

/-- Concrete instances must check every list length used by the interpretation. -/
def LayoutValid (widths : List ℕ) (depth : ℕ) (activations : List Activation)
    (pw pb qw qb : List (List ℝ)) (lower upper budget : List ℝ) : Prop :=
  0 < depth ∧ widths.length = depth + 1 ∧ widths.all (fun w => 0 < w) = true ∧
  activations.length = depth ∧
  pw.length = depth ∧ pb.length = depth ∧ qw.length = depth ∧ qb.length = depth ∧
  (∀ n : Fin depth,
    (pw.getD n.val []).length = widthAt widths n.val * widthAt widths (n.val + 1) ∧
    (qw.getD n.val []).length = widthAt widths n.val * widthAt widths (n.val + 1) ∧
    (pb.getD n.val []).length = widthAt widths (n.val + 1) ∧
    (qb.getD n.val []).length = widthAt widths (n.val + 1)) ∧
  lower.length = widthAt widths 0 ∧ upper.length = widthAt widths 0 ∧
  budget.length = widthAt widths depth

theorem box_drift (widths : List ℕ) (pw pb qw qb : List (List ℝ))
    (activations : List Activation) (lower upper : List ℝ)
    (x : Coord widths 0 → ℝ) (hx : InBox (inputBox widths lower upper) x)
    (n : ℕ) (o : Coord widths n) :
    |evalChain (D := Coord widths) (layers widths qw qb) (activationsAt widths activations) x n o -
      evalChain (D := Coord widths) (layers widths pw pb) (activationsAt widths activations) x n o| ≤
      driftChain (D := Coord widths) (layers widths pw pb) (layers widths qw qb)
        (inputRadius (inputBox widths lower upper)) (fun _ => 0) n o := by
  exact chain_same_input_drift (D := Coord widths) (layers widths pw pb) (layers widths qw qb)
    (activationsAt widths activations) x (inputRadius (inputBox widths lower upper))
    (fun n _ => activation_magnitude (activations.getD n .identity))
    (fun n _ => activation_lipschitz (activations.getD n .identity))
    (abs_input_le_radius (inputBox widths lower upper) x hx) n o

/-- The endpoint identity includes bias separately, in the declared final position. -/
theorem affine_endpoint_exact {F O : Type*} [Fintype F]
    (p q : AffineLayer F O) (x : F → ℝ) (o : O) :
    affineValue q x o - affineValue p x o =
      (q.bias o - p.bias o) + ∑ f, (q.weight o f - p.weight o f) * x f := by
  simp only [affineValue, sub_mul, Finset.sum_sub_distrib]
  ring

end WeightSpace.Bridge
