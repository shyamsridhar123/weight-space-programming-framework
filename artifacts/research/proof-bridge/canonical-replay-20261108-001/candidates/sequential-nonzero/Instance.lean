import WeightSpace.Bridge
import Lean.Util.CollectAxioms

noncomputable section
open scoped BigOperators
namespace WeightSpace.RuntimeInstance
open WeightSpace WeightSpace.Sequential WeightSpace.Bridge
set_option maxRecDepth 200000
set_option maxHeartbeats 2000000

def descriptorHex : String := "7b2262696e64696e67223a6e756c6c2c2263616e646964617465223a7b22617263686974656374757265223a2273657175656e7469616c2d616666696e652d61637469766174696f6e2f31222c226474797065223a22666c6f61743634222c226d65746164617461223a7b22696e7075745f6e616d6573223a5b22666561747572652d61222c22666561747572652d62225d2c226c6179657273223a5b7b2261637469766174696f6e223a2272656c75222c2262696173223a226c61796572732e302e62696173222c22776569676874223a226c61796572732e302e776569676874227d2c7b2261637469766174696f6e223a22616273222c2262696173223a226c61796572732e312e62696173222c22776569676874223a226c61796572732e312e776569676874227d5d2c226f75747075745f6e616d6573223a5b2273636f7265225d7d2c22736368656d61223a227773702d74656e736f72732f31222c2274656e736f7273223a7b226c61796572732e302e62696173223a5b302e302c302e32355d2c226c61796572732e302e776569676874223a5b5b312e3132352c302e355d2c5b302e302c312e305d5d2c226c61796572732e312e62696173223a5b302e303632355d2c226c61796572732e312e776569676874223a5b5b312e302c2d302e3837355d5d7d7d2c2263616e6469646174655f736861323536223a2233633434616465613735616539663433363363313735363831396566346162613133323132376331356439363735613533356437383466383632363239376266222c22646f6d61696e223a7b226c6f776572223a5b2d312e302c2d322e305d2c227570706572223a5b312e302c322e305d7d2c2267656e657261746f725f76657273696f6e223a2231222c2268656164223a6e756c6c2c22696e74657266616365223a7b22616273656e745f626961735f696e746572707265746174696f6e223a227a65726f2d766563746f722d776974682d616273656e63652d72657461696e65642d696e2d68656164222c22626961735f6f72646572223a226578706c696369742d61667465722d646f742d766563746f722f31222c22696e7075745f6e616d6573223a5b22666561747572652d61222c22666561747572652d62225d2c226f75747075745f6e616d6573223a5b2273636f7265225d2c227765696768745f6f72646572223a22726f772d6d616a6f722d6f75747075742d7468656e2d696e7075742f31227d2c226c6179657273223a5b7b2261637469766174696f6e223a2272656c75222c2262696173223a226c61796572732e302e62696173222c22776569676874223a226c61796572732e302e776569676874227d2c7b2261637469766174696f6e223a22616273222c2262696173223a226c61796572732e312e62696173222c22776569676874223a226c61796572732e312e776569676874227d5d2c226f75747075745f6275646765745f6578616374223a5b2231352f3332225d2c22706172656e74223a7b22617263686974656374757265223a2273657175656e7469616c2d616666696e652d61637469766174696f6e2f31222c226474797065223a22666c6f61743634222c226d65746164617461223a7b22696e7075745f6e616d6573223a5b22666561747572652d61222c22666561747572652d62225d2c226c6179657273223a5b7b2261637469766174696f6e223a2272656c75222c2262696173223a226c61796572732e302e62696173222c22776569676874223a226c61796572732e302e776569676874227d2c7b2261637469766174696f6e223a22616273222c2262696173223a226c61796572732e312e62696173222c22776569676874223a226c61796572732e312e776569676874227d5d2c226f75747075745f6e616d6573223a5b2273636f7265225d7d2c22736368656d61223a227773702d74656e736f72732f31222c2274656e736f7273223a7b226c61796572732e302e62696173223a5b302e302c302e32355d2c226c61796572732e302e776569676874223a5b5b312e302c302e355d2c5b302e302c312e305d5d2c226c61796572732e312e62696173223a5b302e305d2c226c61796572732e312e776569676874223a5b5b312e302c2d312e305d5d7d7d2c22706172656e745f736861323536223a2233303964396561383135316566316538663434323661363037326263633135343665303530643163656536623030653736376534313537363732393565353762222c22736368656d61223a227773702d6c65616e2d72756e74696d652d696e7374616e63652f31222c2273636f7065223a2273657175656e7469616c2d616666696e652d61637469766174696f6e227d"
def widths : List ℕ := [2, 2, 1]
def depth : ℕ := 2
def certifiedActivations : List Activation := [.relu, .absolute]
def parentWeights : List (List ℝ) := [[(1 / 1 : ℝ), (1 / 2 : ℝ), (0 / 1 : ℝ), (1 / 1 : ℝ)], [(1 / 1 : ℝ), (-1 / 1 : ℝ)]]
def parentBiases : List (List ℝ) := [[(0 / 1 : ℝ), (1 / 4 : ℝ)], [(0 / 1 : ℝ)]]
def candidateWeights : List (List ℝ) := [[(9 / 8 : ℝ), (1 / 2 : ℝ), (0 / 1 : ℝ), (1 / 1 : ℝ)], [(1 / 1 : ℝ), (-7 / 8 : ℝ)]]
def candidateBiases : List (List ℝ) := [[(0 / 1 : ℝ), (1 / 4 : ℝ)], [(1 / 16 : ℝ)]]
def domainLower : List ℝ := [(-1 / 1 : ℝ), (-2 / 1 : ℝ)]
def domainUpper : List ℝ := [(1 / 1 : ℝ), (2 / 1 : ℝ)]
def outputBudget : List ℝ := [(15 / 32 : ℝ)]

theorem layout_valid : LayoutValid widths depth certifiedActivations parentWeights parentBiases
    candidateWeights candidateBiases domainLower domainUpper outputBudget := by
  unfold LayoutValid
  decide

theorem domain_valid : (inputBox widths domainLower domainUpper).Valid := by
  change ∀ i : Fin 2, _
  norm_num [inputBox, widthAt, widths, domainLower, domainUpper, Fin.forall_fin_succ]

theorem numeric_budget : ∀ o : Coord widths depth,
    driftChain (D := Coord widths) (layers widths parentWeights parentBiases)
      (layers widths candidateWeights candidateBiases)
      (inputRadius (inputBox widths domainLower domainUpper)) (fun _ => 0) depth o ≤
      outputBudget.getD o.val 0 := by
  change ∀ o : Fin 1, _
  dsimp only [Coord, widthAt, widths, depth, driftChain, driftStep, radiusChain,
    radiusStep, layers, inputRadius, inputBox, parentWeights, parentBiases,
    candidateWeights, candidateBiases, domainLower, domainUpper, outputBudget]
  norm_num [Fin.sum_univ_succ, Fin.forall_fin_succ]

theorem certified_output_drift (x : Coord widths 0 → ℝ)
    (hx : InBox (inputBox widths domainLower domainUpper) x) (o : Coord widths depth) :
    |evalChain (D := Coord widths) (layers widths candidateWeights candidateBiases)
       (activationsAt widths certifiedActivations) x depth o -
     evalChain (D := Coord widths) (layers widths parentWeights parentBiases)
       (activationsAt widths certifiedActivations) x depth o| ≤ outputBudget.getD o.val 0 := by
  exact (box_drift widths parentWeights parentBiases candidateWeights candidateBiases
    certifiedActivations domainLower domainUpper x hx depth o).trans (numeric_budget o)

theorem affine_endpoint_identity (n : Fin depth) (x : Coord widths n.val → ℝ)
    (o : Coord widths (n.val + 1)) :
    affineValue (layers widths candidateWeights candidateBiases n.val) x o -
      affineValue (layers widths parentWeights parentBiases n.val) x o =
      ((layers widths candidateWeights candidateBiases n.val).bias o -
        (layers widths parentWeights parentBiases n.val).bias o) +
      ∑ i, ((layers widths candidateWeights candidateBiases n.val).weight o i -
        (layers widths parentWeights parentBiases n.val).weight o i) * x i := by
  exact affine_endpoint_exact _ _ x o

#print axioms certified_output_drift
#print axioms affine_endpoint_identity

run_cmd do
  let allowed : Array Lean.Name := #[``propext, ``Classical.choice, ``Quot.sound]
  let env ← Lean.getEnv
  let checked : Array Lean.Name := env.constants.fold (fun names name _ =>
    if (`WeightSpace.Bridge).isPrefixOf name || (`WeightSpace.RuntimeInstance).isPrefixOf name
    then names.push name else names) #[]
  for decl in checked do
    for axiomName in (← Lean.collectAxioms decl) do
      unless allowed.contains axiomName do
        throwError "{decl} depends on disallowed axiom {axiomName}"
  Lean.logInfo m!"Runtime instance transitive axiom audit passed."

end WeightSpace.RuntimeInstance
