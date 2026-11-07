# An outage. The right team.

A 90-second visual explanation of the Weight Space Programming Framework through enterprise support routing.

The subtitles below are captured in the browser. There is no spoken narrator. The music is a separate quiet instrumental track.

**00:00–00:05** An outage ticket lands with the wrong team.

**00:05–00:10** The policy should send it straight to Platform Operations.

**00:10–00:15** The router gives each team a score. The highest score wins.

**00:15–00:20** For this ticket, General Support scores 2. Platform scores only 0.25.

**00:20–00:25** A weight sets how much a feature contributes to a score.

**00:25–00:30** The outage feature is on here, but off for ordinary support.

**00:30–00:35** A target-only edit improves the outage route.

**00:35–00:40** But it also changes an ordinary ticket’s scores. That fails the protection rule.

**00:40–00:45** Think of an edit as a direction in weight space.

**00:45–00:51** Choose a direction with zero effect on the protected inputs.

**00:51–00:56** Here, the framework changes only the outage-to-Platform weight: 0.25 becomes about 3.22.

**00:56–01:01** Now Platform wins this ticket. The model changes through an explicit, checked weight edit.

**01:01–01:06** Final recognized outages reach Platform. The sampled protected scores stay unchanged.

**01:06–01:11** A direct route could avoid a transfer. Handling time and savings have not been measured.

**01:11–01:16** If a target must also stay fixed, the framework abstains.

**01:16–01:21** If a new symptom is not recognized, the final check rejects the candidate.

**01:21–01:25** The bound inverse restores the exact parent model. It cannot undo ticket assignments.

**01:25–01:30** The method is generic. This demonstration uses fixed features and a supplied linear model.

## Evidence and limits

All scores and edits come from the scenario bundles linked in [the source register](source.json). The bar chart follows one public standard outage record. The final result summarizes the declared synthetic final split. A possible avoided handoff is an operational opportunity, not a measured business outcome.

The two-dimensional weight picture is a schematic slice. The compiler handles the remaining feature and bias constraints. Its numerical fit uses ridge regularization, so the target shift is approximate. The exact retention certificate covers the supplied fixed-feature anchors and their span under real-valued linear semantics; it does not prove floating-point execution or text understanding.

The interactive page starts paused. Use Play, chapter arrows, the time slider, and Reset. The local server recomputes all scenarios and applies the inverse before playback. A portable static page uses the recorded, verified source model. Readable prose and the understanding check remain available without JavaScript.

The explanation follows STE-Pro-Max's concrete-case-first storytelling and visual guidance at commit `a7da74e83e5fcf8ea23dce075e64b463e5e24584`. Its figures and code are original. No Mermaid is used.
