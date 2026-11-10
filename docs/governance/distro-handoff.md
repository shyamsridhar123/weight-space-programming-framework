# Optional Distro authoring and governance handoff

**Scope: November 10, 2026.** A support owner wants telemetry outage tickets
routed to Platform Operations while preserving billing, security and ordinary
support behavior. An authoring tool can help assemble that request, a proposed
implementation and its review packet. The organization's authority to approve
and operate the change is a separate obligation.

[Distro's official page](https://delta.microsoft.com/distro), rendered on
November 10, 2026, describes a coding-agent plugin that bundles customer context,
examples, tools, connections and skills, with a workflow for prototypes, review
and publishing. The page says Distro does not replace the coding agent or
complete a production handoff. This supports considering Distro as an optional
authoring and handoff tool; it does not establish a verified governance
integration. No Distro governance API, authentication mechanism, enforced
separation of roles, production audit
facility or license suitability has been verified for this pilot. Do not infer
those capabilities from its authoring workflow.

The recommended first step is the generic local
[governance contract](contract.md), using synthetic records. It has no Distro
dependency or installation requirement. A Distro-assisted draft, a manually
prepared draft and a draft from another tool should enter the same review
boundary. This document proposes a handoff; it does not report a deployed
integration, an authenticated organizational approval or a completed production
change.

## Establish what each evidence layer means

The governance decision must keep four kinds of evidence distinct.

| Layer | What it can support | What it leaves open |
| --- | --- | --- |
| Conditional formal invariants | A checked theorem can establish a stated property of an explicit transition model under its premises. | The implementation's correspondence to that model, truthful inputs, authenticated people and real production actions need separate evidence. A proposed theorem or unexecuted proof file is not a successful proof run. |
| Synthetic implementation checks | A recorded check can establish the declared behavior on the exercised local records and code paths. | Synthetic actors, signatures or approval events do not establish an organization's identity provider, role assignments, enforcement or production audit. Read the actual check receipt before claiming a pass. |
| Authenticated organizational governance | Independently authenticated authority, policy, approvals and operational receipts can support a bounded organizational decision. | The local pilot does not supply these integrations or authenticate a caller's declarations. A locally self-consistent history alone is insufficient. |
| Scientific and model evidence | A frozen study and its retained outcomes can support its specified model, task and population claims. | An approved change, a formal workflow invariant or a passing software check does not establish useful editing, generalization, complete costs or business benefit. |

For the example, the
[enterprise support-routing walkthrough](../use-cases/enterprise-support-routing.md)
provides a synthetic integration pattern. The
[live studies](../research/live-studies.md) retained 80 primary weight-access
abstentions and diagnosis coverage below its frozen threshold. The
[original-source repeat](../research/original-source-repeat.md) retained four
failed calls with no completed response, and the
[construction and steering pilot](../research/construction-steering-pilot.md)
retained twelve failed live assignments with no candidate to score. Successful
controls, evidence packaging and replay preserve their own meaning; they do not
supply live-model usefulness or customer savings. These negative outcomes must
remain visible when an authoring tool summarizes the project.

## Prepare one reviewable change

1. **Name the use case and its owner.** Record the service owner, affected
   users, intended benefit, scope, prohibited uses and who can accept the
   resulting operational risk. For this example, define an outage ticket,
   Platform Operations ownership and the protected ticket classes. A claimed
   reduction in transfers or waiting time needs its own observation and metric.

2. **Freeze the proposed change.** Bind the exact parent, candidate or patch,
   configuration, relevant dependencies and intended environment. Include a
   readable difference and the requested behavioral contract. Declare whether
   the proposal changes a routing policy, model weights, a prompt, an adapter
   or a combination. A changed artifact or scope needs a new review; it must
   not inherit approval through an unchanged display name.

3. **Record the author's information and access.** Inventory the context,
   examples, tools, connections and skills actually used. Identify their
   sources, versions, permissions and data roles, including any human help.
   A Distro connection must not be treated as permission to read enterprise
   data. SharePoint content, email and meeting records require explicit
   authorization for the relevant accounts, purposes and records before
   retrieval or transfer. Prepare the local pilot with synthetic material.

4. **Attach evidence with its limits.** Bind the change to the exact checks,
   protocols, source versions, artifacts and receipts offered for review.
   Separate structural validity, finite behavior, formal premises, access
   enforcement, cost and operational impact. Retain rejected candidates,
   abstentions, delivery failures, unknown quantities and the full assigned
   denominator. Observed lower bounds and missing telemetry must not become
   zero-cost or successful observations in the handoff.

5. **Obtain a review by an authorized person.** Give the reviewer the frozen
   use case, change, evidence and unresolved checks. Record the reviewer's
   authenticated identity and authority outside the synthetic pilot, the
   decision, reasons, conditions, scope and expiration. Establish the required
   author/reviewer/approver separation in the organization's policy and enforce
   it in its actual systems. Self-declared actor names in local records cannot
   establish that separation. Material revisions require another review.

6. **Approve a specific operational action.** The authorized approver must
   identify the reviewed artifact, destination, allowed action, conditions and
   validity period. Preserve an explicit rejection or pending decision where
   evidence is inadequate. A local pilot record or Distro's publish action
   cannot substitute for authorization to deploy, distribute or expose data.
   Production execution needs its own controlled mechanism and receipt bound
   to the approved artifact and environment.

7. **Retain monitoring and revocation responsibilities.** Name who monitors
   target and protected behavior, the stop conditions, and who can revoke
   approval. A revocation should preserve the original decision and its reason,
   prevent further use under that grant, and trigger the organization's
   withdrawal or rollback process. Confirm the actual operational state
   separately: recording revocation locally does not undo a deployment, remove
   distributed copies or revoke third-party credentials. Preserve earlier
   evidence and issue a new review for any replacement change.

The handoff packet should let a reviewer locate each of those records without
opening unrelated customer material. Use the local contract's actual record
format and supported transitions. This proposal defines no Distro endpoint,
SDK call, webhook or token format.

## Resolve access, cost, data and reuse before an enterprise pilot

The following checks are unresolved. Their owners must record an explicit
disposition before relying on an enterprise integration; a product description
does not discharge them.

| Check | Evidence needed for the proposed scope |
| --- | --- |
| Connection access | Approved accounts, resources, purposes and operations; least necessary permissions; connector identity; credential storage, expiration and revocation; evidence that the actual connection enforces those limits. |
| Data handling | Allowed data classes, residency, retention, deletion, subprocessors, model-training or secondary-use terms, and the handling of prompts, examples, logs, attachments and exported packets. Establish redaction and export permissions before transfer. |
| Role and authority separation | An authenticated actor source, authoritative role assignments, required independent decisions, enforcement against bypass, and treatment of role removal or expired authority. |
| Audit and production handoff | Durable evidence bound to exact artifacts and policy versions, trusted time and retention where required, external commitments if tamper detection is claimed, and a controlled deployment and revocation path. Distro support for these requirements is unverified. |
| Complete cost | Subscription and connector charges, provider usage including failed requests and retries, preparation, human review, evaluation, storage and operations. State currency, time period, accounting boundary and unknown amounts; compare only matched quality and scope. |
| Rights and permitted reuse | Distro and plugin terms, dependency licenses, customer-content rights and permission for copied examples, generated artifacts and later reuse. Repository licensing alone does not establish suitability for customer data or an enterprise deployment. |
| Evaluation reuse | Which records were used to author or select the change, which remain evaluation-only, permitted feedback and future reuse. Preserve observed lineage and consumption; a connection or record hash cannot establish semantic independence or freshness. |
| Operational value | Frozen target and retention criteria, the independent units for a confirmation, monitoring, stop conditions and rollback ownership. Synthetic routing success and local approvals are insufficient evidence of customer benefit. |

A local synthetic pilot can make the review record and its failure cases
inspectable while those checks remain open. It cannot grant enterprise access,
complete a vendor assessment or change the scientific evidence. Any later
integration should retain this pilot's source and results as a separate record
and report its own verified scope.
