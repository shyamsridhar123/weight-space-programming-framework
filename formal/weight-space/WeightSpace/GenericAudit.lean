import WeightSpace
import WeightSpace.GenericExamples
import Lean.Util.CollectAxioms

/-!
# V2 transitive axiom audit

This separate entry point preserves the original library root and old evidence.
It loads both the original library and every v2 module, audits every WeightSpace
declaration, and separately reports the new namespaces. The audit rejects
placeholders, added axioms, and native-evaluation proof shortcuts transitively.
-/

#print axioms WeightSpace.Generic.edit_exact
#print axioms WeightSpace.Generic.endpoint_exact
#print axioms WeightSpace.Generic.feature_span_effect
#print axioms WeightSpace.Generic.protected_span
#print axioms WeightSpace.Generic.residual_span_bound
#print axioms WeightSpace.Generic.residual_span_l1_bound
#print axioms WeightSpace.Generic.endpoint_span_bound
#print axioms WeightSpace.Generic.approximate_span_bound
#print axioms WeightSpace.Generic.nonzero_target_in_protected_span_impossible
#print axioms WeightSpace.Generic.factored_effect
#print axioms WeightSpace.Generic.factored_preservation
#print axioms WeightSpace.Generic.factored_span_preservation
#print axioms WeightSpace.Generic.factored_residual_bound
#print axioms WeightSpace.Sequential.layer_radius_le
#print axioms WeightSpace.Sequential.layer_drift_le
#print axioms WeightSpace.Sequential.chain_radius_le
#print axioms WeightSpace.Sequential.chain_drift_le
#print axioms WeightSpace.Sequential.chain_same_input_drift
#print axioms WeightSpace.Sequential.coordinate_drift_compose
#print axioms WeightSpace.GenericExamples.nonlinear_feature_protected_line
#print axioms WeightSpace.GenericExamples.target_output_zero_change
#print axioms WeightSpace.GenericExamples.target_output_one_change
#print axioms WeightSpace.GenericExamples.extrapolation_l1_certificate
#print axioms WeightSpace.GenericExamples.extrapolation_l1_bound_is_attained
#print axioms WeightSpace.GenericExamples.factored_nonlinear_protection
#print axioms WeightSpace.GenericExamples.protected_target_one_infeasible
#print axioms WeightSpace.GenericExamples.three_layer_uniform_certificate
#print axioms WeightSpace.GenericExamples.three_layer_endpoint_difference

run_cmd do
  let allowed : Array Lean.Name := #[``propext, ``Classical.choice, ``Quot.sound]
  let env ← Lean.getEnv
  let checked : Array Lean.Name := env.constants.fold (fun names name _ =>
    if (`WeightSpace).isPrefixOf name then names.push name else names) #[]
  let newNamespaces := #[`WeightSpace.Generic, `WeightSpace.Sequential, `WeightSpace.GenericExamples]
  let added := checked.filter (fun name => newNamespaces.any (fun ns => ns.isPrefixOf name))
  let countTheorems (names : Array Lean.Name) : Nat := names.foldl (fun count name =>
    match env.find? name with
    | some (.thmInfo _) => count + 1
    | _ => count) 0
  let mut observed : Array Lean.Name := #[]
  for decl in checked do
    for axiomName in (← Lean.collectAxioms decl) do
      unless allowed.contains axiomName do
        throwError "{decl} depends on disallowed axiom {axiomName}"
      unless observed.contains axiomName do
        observed := observed.push axiomName
  Lean.logInfo m!"V2 full transitive axiom audit passed: {checked.size} declarations; {countTheorems checked} theorem declarations."
  Lean.logInfo m!"V2 new namespaces: {added.size} declarations; {countTheorems added} theorem declarations."
  for ns in newNamespaces do
    let names := added.filter (fun name => ns.isPrefixOf name)
    Lean.logInfo m!"V2 namespace {ns}: {names.size} declarations; {countTheorems names} theorem declarations."
  Lean.logInfo m!"V2 observed axioms: {observed}."
