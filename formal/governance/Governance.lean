import Std
import Lean.Elab.Command
import Lean.Util.CollectAxioms

/-!
An abstract governance state machine for the optional local pilot.

This file does not model JSON, SHA-256, files, authentication, deployment,
scientific validity, or the Python implementation. Encoding functions are
arbitrary parameters; no collision-resistance or injectivity axiom is used.
Expected pins are explicit inputs. Their independent origin and freshness
cannot be established by this state machine. See README.md for verification
status and the remaining correspondence assumptions.
-/

namespace Governance

abbrev Actor := String
abbrev Digest := String

inductive Environment where
  | localPilot
  | production
  | scientific

structure Scope where
  environment : Environment
  useCase : String
  target : String

structure Context where
  base : Digest
  program : Digest
  candidate : Digest
  constraints : Digest
  scope : Scope

structure Request where
  context : Context
  purpose : String
  evidence : List Digest
  requester : Actor
  approvers : List Actor

/-- An abstract record of the approval retained in the validated history. -/
structure Approval where
  actor : Actor
  requestHash : Digest

inductive Status where
  | requested
  | approved (approval : Approval)
  | rejected (actor : Actor)
  | revoked (actor : Actor)

structure State where
  request : Request
  status : Status
  head : Digest

inductive Action where
  | approve
  | reject
  | revoke

structure Decision where
  action : Action
  actor : Actor
  requestHash : Digest
  previousHead : Digest

/-- These functions abstract the implementation's encoding and hashing. -/
structure Encoding where
  requestHash : Request → Digest
  initialHead : Request → Digest
  decisionHead : Decision → Digest

/-- Both values must be supplied by the caller; there are no defaults. -/
structure ExpectedPins where
  requestHash : Digest
  head : Digest

structure PinsMatch (encoding : Encoding) (expected : ExpectedPins)
    (state : State) : Prop where
  requestHash : expected.requestHash = encoding.requestHash state.request
  head : expected.head = state.head

structure Bound (encoding : Encoding) (expected : ExpectedPins)
    (before : State) (decision : Decision) : Prop where
  pins : PinsMatch encoding expected before
  requestHash : decision.requestHash = encoding.requestHash before.request
  previousHead : decision.previousHead = before.head

/-- Every constructor preserves the entire request, including its scope. -/
inductive Step (encoding : Encoding) (expected : ExpectedPins) :
    State → Decision → State → Prop where
  | approve {before : State} {decision : Decision}
      (requested : before.status = .requested)
      (bound : Bound encoding expected before decision)
      (action : decision.action = .approve)
      (allowed : decision.actor ∈ before.request.approvers)
      (separate : decision.actor ≠ before.request.requester) :
      Step encoding expected before decision
        { before with
          status := .approved ⟨decision.actor, decision.requestHash⟩
          head := encoding.decisionHead decision }
  | reject {before : State} {decision : Decision}
      (requested : before.status = .requested)
      (bound : Bound encoding expected before decision)
      (action : decision.action = .reject)
      (allowed : decision.actor ∈ before.request.approvers)
      (separate : decision.actor ≠ before.request.requester) :
      Step encoding expected before decision
        { before with
          status := .rejected decision.actor
          head := encoding.decisionHead decision }
  | revoke {before : State} {decision : Decision}
      (approved : ∃ approval, before.status = .approved approval)
      (bound : Bound encoding expected before decision)
      (action : decision.action = .revoke)
      (allowed : decision.actor = before.request.requester ∨
        decision.actor ∈ before.request.approvers) :
      Step encoding expected before decision
        { before with
          status := .revoked decision.actor
          head := encoding.decisionHead decision }

def initial (encoding : Encoding) (request : Request) : State :=
  { request := request
    status := .requested
    head := encoding.initialHead request }

inductive Path (encoding : Encoding) (start : State) : State → Prop where
  | refl : Path encoding start start
  | extend {current next : State} {expected : ExpectedPins}
      {decision : Decision}
      (prefix : Path encoding start current)
      (transition : Step encoding expected current decision next) :
      Path encoding start next

def Reachable (encoding : Encoding) (request : Request) (state : State) : Prop :=
  Path encoding (initial encoding request) state

def ApprovalInvariant (encoding : Encoding) (state : State) : Prop :=
  ∀ approval, state.status = .approved approval →
    approval.actor ∈ state.request.approvers ∧
    approval.actor ≠ state.request.requester ∧
    approval.requestHash = encoding.requestHash state.request

def Terminal (state : State) : Prop :=
  match state.status with
  | .rejected _ => True
  | .revoked _ => True
  | .requested => False
  | .approved _ => False

theorem step_preserves_request
    {encoding : Encoding} {expected : ExpectedPins}
    {before after : State} {decision : Decision}
    (transition : Step encoding expected before decision after) :
    after.request = before.request := by
  cases transition <;> rfl

theorem step_is_bound
    {encoding : Encoding} {expected : ExpectedPins}
    {before after : State} {decision : Decision}
    (transition : Step encoding expected before decision after) :
    Bound encoding expected before decision := by
  cases transition <;> assumption

theorem mismatched_pins_cannot_step
    {encoding : Encoding} {expected : ExpectedPins}
    {before after : State} {decision : Decision}
    (mismatch : expected.requestHash ≠ encoding.requestHash before.request ∨
      expected.head ≠ before.head) :
    ¬ Step encoding expected before decision after := by
  intro transition
  have bound := step_is_bound transition
  cases mismatch with
  | inl wrongRequest => exact wrongRequest bound.pins.requestHash
  | inr wrongHead => exact wrongHead bound.pins.head

theorem path_preserves_request
    {encoding : Encoding} {start finish : State}
    (path : Path encoding start finish) :
    finish.request = start.request := by
  induction path with
  | refl => rfl
  | extend prefix transition inductionHypothesis =>
      exact (step_preserves_request transition).trans inductionHypothesis

theorem reachable_preserves_request
    {encoding : Encoding} {request : Request} {state : State}
    (reachable : Reachable encoding request state) :
    state.request = request := by
  exact path_preserves_request reachable

theorem reachable_step_binds_original
    {encoding : Encoding} {request : Request} {expected : ExpectedPins}
    {before after : State} {decision : Decision}
    (reachable : Reachable encoding request before)
    (transition : Step encoding expected before decision after) :
    expected.requestHash = encoding.requestHash request ∧
    expected.head = before.head ∧
    decision.requestHash = encoding.requestHash request ∧
    decision.previousHead = before.head := by
  have bound := step_is_bound transition
  have preserved := reachable_preserves_request reachable
  refine ⟨?_, bound.pins.head, ?_, bound.previousHead⟩
  · simpa only [preserved] using bound.pins.requestHash
  · simpa only [preserved] using bound.requestHash

theorem step_approval_is_separate_and_bound
    {encoding : Encoding} {expected : ExpectedPins}
    {before after : State} {decision : Decision}
    (transition : Step encoding expected before decision after) :
    ApprovalInvariant encoding after := by
  cases transition with
  | approve requested bound action allowed separate =>
      intro approval approved
      cases Status.approved.inj approved
      exact ⟨allowed, separate, bound.requestHash⟩
  | reject =>
      intro approval approved
      cases approved
  | revoke =>
      intro approval approved
      cases approved

theorem reachable_approval_invariant
    {encoding : Encoding} {request : Request} {state : State}
    (reachable : Reachable encoding request state) :
    ApprovalInvariant encoding state := by
  change Path encoding (initial encoding request) state at reachable
  induction reachable with
  | refl =>
      intro approval approved
      change Status.requested = Status.approved approval at approved
      cases approved
  | extend prefix transition inductionHypothesis =>
      exact step_approval_is_separate_and_bound transition

theorem reachable_approval_is_separate_and_bound
    {encoding : Encoding} {request : Request} {state : State}
    (reachable : Reachable encoding request state)
    (approval : Approval) (approved : state.status = .approved approval) :
    approval.actor ∈ request.approvers ∧
    approval.actor ≠ request.requester ∧
    approval.requestHash = encoding.requestHash request := by
  have valid := reachable_approval_invariant reachable approval approved
  rw [reachable_preserves_request reachable] at valid
  exact valid

theorem terminal_no_step
    {encoding : Encoding} {expected : ExpectedPins}
    {before after : State} {decision : Decision}
    (terminal : Terminal before)
    (transition : Step encoding expected before decision after) : False := by
  cases transition with
  | approve requested bound action allowed separate =>
      simp [Terminal, requested] at terminal
  | reject requested bound action allowed separate =>
      simp [Terminal, requested] at terminal
  | revoke approved bound action allowed =>
      cases approved with
      | intro approval status => simp [Terminal, status] at terminal

theorem terminal_path_stays
    {encoding : Encoding} {start finish : State}
    (terminal : Terminal start) (path : Path encoding start finish) :
    finish = start := by
  induction path with
  | refl => rfl
  | extend prefix transition inductionHypothesis =>
      rw [inductionHypothesis] at transition
      exact False.elim (terminal_no_step terminal transition)

theorem revoked_path_stays
    {encoding : Encoding} {start finish : State} {actor : Actor}
    (revoked : start.status = .revoked actor)
    (path : Path encoding start finish) :
    finish = start := by
  have terminal : Terminal start := by
    simp [Terminal, revoked]
  exact terminal_path_stays terminal path

/-- Authorization requires replayable history, both pins, and exact context. -/
structure Authorized (encoding : Encoding) (request : Request)
    (expected : ExpectedPins) (state : State) (current : Context)
    (use : Environment) : Prop where
  reachable : Reachable encoding request state
  pins : PinsMatch encoding expected state
  context : current = state.request.context
  scope : current.scope.environment = .localPilot
  approved : ∃ approval, state.status = .approved approval
  localOnly : use = .localPilot

theorem authorized_pins_and_context
    {encoding : Encoding} {request : Request} {expected : ExpectedPins}
    {state : State} {current : Context} {use : Environment}
    (authorization : Authorized encoding request expected state current use) :
    expected.requestHash = encoding.requestHash request ∧
    expected.head = state.head ∧
    current = request.context := by
  have preserved := reachable_preserves_request authorization.reachable
  refine ⟨?_, authorization.pins.head, ?_⟩
  · simpa only [preserved] using authorization.pins.requestHash
  · simpa only [preserved] using authorization.context

theorem authorized_approval_is_separate_and_bound
    {encoding : Encoding} {request : Request} {expected : ExpectedPins}
    {state : State} {current : Context} {use : Environment}
    (authorization : Authorized encoding request expected state current use) :
    ∃ approval, state.status = .approved approval ∧
      approval.actor ∈ request.approvers ∧
      approval.actor ≠ request.requester ∧
      approval.requestHash = encoding.requestHash request := by
  cases authorization.approved with
  | intro approval approved =>
      exact ⟨approval, approved,
        reachable_approval_is_separate_and_bound
          authorization.reachable approval approved⟩

theorem authorized_is_local
    {encoding : Encoding} {request : Request} {expected : ExpectedPins}
    {state : State} {current : Context} {use : Environment}
    (authorization : Authorized encoding request expected state current use) :
    use = .localPilot := by
  exact authorization.localOnly

theorem production_authorization_impossible
    {encoding : Encoding} {request : Request} {expected : ExpectedPins}
    {state : State} {current : Context} :
    ¬ Authorized encoding request expected state current .production := by
  intro authorization
  cases authorization.localOnly

theorem scientific_authorization_impossible
    {encoding : Encoding} {request : Request} {expected : ExpectedPins}
    {state : State} {current : Context} :
    ¬ Authorized encoding request expected state current .scientific := by
  intro authorization
  cases authorization.localOnly

theorem terminal_cannot_authorize
    {encoding : Encoding} {request : Request} {expected : ExpectedPins}
    {state : State} {current : Context} {use : Environment}
    (terminal : Terminal state) :
    ¬ Authorized encoding request expected state current use := by
  intro authorization
  cases authorization.approved with
  | intro approval status => simp [Terminal, status] at terminal

#print axioms step_preserves_request
#print axioms step_is_bound
#print axioms mismatched_pins_cannot_step
#print axioms path_preserves_request
#print axioms reachable_preserves_request
#print axioms reachable_step_binds_original
#print axioms step_approval_is_separate_and_bound
#print axioms reachable_approval_invariant
#print axioms reachable_approval_is_separate_and_bound
#print axioms terminal_no_step
#print axioms terminal_path_stays
#print axioms revoked_path_stays
#print axioms authorized_pins_and_context
#print axioms authorized_approval_is_separate_and_bound
#print axioms authorized_is_local
#print axioms production_authorization_impossible
#print axioms scientific_authorization_impossible
#print axioms terminal_cannot_authorize

end Governance

-- Keep this audit after every declaration in the Governance namespace.
run_cmd do
  let allowed : Array Lean.Name := #[``propext, ``Classical.choice, ``Quot.sound]
  let env ← Lean.getEnv
  let declarations : Array Lean.Name := env.constants.fold (fun names name _ =>
    if (`Governance).isPrefixOf (Lean.privateToUserName name) then
      names.push name else names) #[]
  if declarations.isEmpty then
    throwError "No Governance declarations found for axiom audit"
  for declaration in declarations do
    for axiomName in (← Lean.collectAxioms declaration) do
      unless allowed.contains axiomName do
        throwError "{declaration} depends on disallowed axiom {axiomName}"
  Lean.logInfo m!"Governance transitive axiom audit passed."
