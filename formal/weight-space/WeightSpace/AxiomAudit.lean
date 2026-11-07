import WeightSpace.Core
import WeightSpace.Gauge
import WeightSpace.Bounds
import WeightSpace.Contracts
import WeightSpace.Examples
import WeightSpace.RTS
import Lean.Util.CollectAxioms

/-!
These commands make the transitive trust dependencies visible in build output.
The project adds no axioms to Lean. Standard real analysis in Mathlib uses
propositional extensionality, choice, and quotient soundness.
-/

#print axioms WeightSpace.score_permute
#print axioms WeightSpace.transported_edit_score
#print axioms WeightSpace.score_rescale
#print axioms WeightSpace.rescaled_edit_score
#print axioms WeightSpace.readout_exact
#print axioms WeightSpace.readoutEffect_add
#print axioms WeightSpace.editReadout_add
#print axioms WeightSpace.relu_lipschitz
#print axioms WeightSpace.score_drift_le
#print axioms WeightSpace.score_drift_le_of_budget
#print axioms WeightSpace.readout_drift_le
#print axioms WeightSpace.readout_single_drift_le
#print axioms WeightSpace.readout_single_drift_le_of_budget
#print axioms WeightSpace.DriftContract.trans
#print axioms WeightSpace.DriftContract.chain
#print axioms WeightSpace.argmax_preserved
#print axioms WeightSpace.argmax_preserved_uniform
#print axioms WeightSpace.Examples.all_layer_certificate
#print axioms WeightSpace.Examples.readout_certificate
#print axioms WeightSpace.Examples.protected_region
#print axioms WeightSpace.RTS.recruit_0_5_exact
#print axioms WeightSpace.RTS.recruit_0_5_bound
#print axioms WeightSpace.RTS.recruit_0_5_protected
#print axioms WeightSpace.RTS.recruit_1_0_exact
#print axioms WeightSpace.RTS.recruit_1_0_bound
#print axioms WeightSpace.RTS.recruit_1_0_protected
#print axioms WeightSpace.RTS.recruit_2_0_exact
#print axioms WeightSpace.RTS.recruit_2_0_bound
#print axioms WeightSpace.RTS.recruit_2_0_protected
#print axioms WeightSpace.RTS.recruit_4_0_exact
#print axioms WeightSpace.RTS.recruit_4_0_bound
#print axioms WeightSpace.RTS.recruit_4_0_protected

/- Fail elaboration if any declaration in the formal namespace depends on a
    proof placeholder, an evaluation shortcut, or an additional axiom. -/
run_cmd do
  let allowed : Array Lean.Name := #[``propext, ``Classical.choice, ``Quot.sound]
  let env ← Lean.getEnv
  let checked : Array Lean.Name := env.constants.fold (fun names name _ =>
    if (`WeightSpace).isPrefixOf name then names.push name else names) #[]
  let rts := checked.filter (fun name => (`WeightSpace.RTS).isPrefixOf name)
  let countTheorems (names : Array Lean.Name) : Nat := names.foldl (fun count name =>
    match env.find? name with
    | some (.thmInfo _) => count + 1
    | _ => count) 0
  for decl in checked do
    for axiomName in (← Lean.collectAxioms decl) do
      unless allowed.contains axiomName do
        throwError "{decl} depends on disallowed axiom {axiomName}"
  Lean.logInfo m!"Axiom audit passed for {checked.size} declarations."
  Lean.logInfo m!"Theorem declarations: {countTheorems checked}; RTS declarations: {rts.size}; RTS theorem declarations: {countTheorems rts}."
