# README artwork brief

**Status: planned artwork; no ChatGPT-generated assets exist in this directory.**

The README draft uses the existing walkthrough poster, framework diagram, and
edit-geometry illustration. The images below are the intended replacements and
additions. Their filenames are plans, not links to completed files.

On November 10, 2026, two built-in ChatGPT image-generation attempts failed before
producing an image: HTTP 404, with the configured provider reported as unavailable.
No API fallback was used. This brief preserves the prompts so generation can
resume without changing the technical story.

## Shared art direction

Use a coordinated editorial style: warm ivory, charcoal, muted teal, and small
terracotta or brass accents. Favor generous space, readable type, fine linework,
and restrained depth. Diagram text must remain readable at GitHub README width;
essential relationships must also be explained in the surrounding Markdown.

Avoid neon, robots, brains, generic cloud icons, fake metrics, invented labels,
and decorative arrows. No Mermaid. Illustrations are conceptual explanations,
not measured plots, captured interfaces, or proof evidence. Color should support
text and shape, never carry meaning alone.

Use the built-in ChatGPT image tool for each asset. The dimensions below describe
the desired composition, not a guarantee of tool output dimensions. After each
generation, inspect spelling, arrows, technical relationships, and readability
before copying the selected output into this directory. Record its actual
dimensions, byte hash, tool, prompt, and any edits in a separate provenance record.
Do not attribute generated artwork to the historical flowchart renderer.

## 1. Hero

Planned filename: `hero.png`

Desired composition: approximately 2400 × 900 pixels.

Alt text: “Weight Space Programming Framework: an inspectable change to a weight
matrix beside explicitly protected directions. Read, diagnose, edit, create.”

```text
Use case: illustration-story.
Asset type: a premium editorial hero banner for the public GitHub README of
"Weight Space Programming Framework".

Create ONE finished landscape image, very wide, approximately 2400 by 900 pixels
(8:3 aspect ratio). This is a polished, restrained research-software cover, with
the richness of a commissioned scientific magazine illustration. Warm ivory
paper background, deep charcoal ink, muted jade/teal, a very small amount of
terracotta and brass. Fine grain, soft studio shadows, abundant breathing room.
No neon, no sci-fi tunnel, no robot, no generic brain, no stock cloud icons, no
badges or fake metrics.

Left 48 percent: beautifully typeset large editorial serif title, exactly three
lines:
"Weight Space"
"Programming"
"Framework"

Below in a small crisp sans-serif, exactly:
"Read. Diagnose. Edit. Create."

The left text is flat ink, razor-sharp, exceptionally readable. Preserve generous
margins.

Right 52 percent: a considered sculptural illustration of a small neural weight
matrix as a 5 by 5 arrangement of short upright porcelain tiles on a slightly
tilted square slab. Most tiles are ivory with delicate charcoal coordinate
lines. Exactly one central tile is terracotta, representing a literal changed
weight. A thin teal plane or ribbon passes beside the matrix and continues
smoothly beyond it, representing explicitly protected feature directions. A
second fine terracotta trajectory bends at the single changed tile. Include a
thin technical annotation line leading to that tile labelled exactly
"an inspectable edit". Another thin line to the teal ribbon labelled exactly
"explicit constraints". Both labels small but legible.

This is a conceptual illustration, not a measured plot or simulation. No tick
marks or invented numeric values. The object has elegant spatial depth,
realistic matte ceramic and translucent vellum materials, restrained warm
lighting, and fine diagram-like linework. A subtle rectangular inset outline may
frame the matrix without containing the left text. Only the specified words
appear. No watermarks, no logos, no decorative gibberish. Keep all essential
content inset from image boundaries.
```

## 2. Generic framework architecture

Planned filename: `framework-architecture.png`

Desired composition: approximately 2400 × 1500 pixels.

Alt text: “A task contract branches into read or diagnose answers, edit proposals
lowered to actual candidates, and complete created artifacts. Session history
feeds explicit caller-run checks, selection or abstention, and sealing. Selected
candidates reach caller-owned held-out evaluation; abstention reaches a receipt
without an evaluator call.”

```text
Use case: infographic-diagram.
Asset type: the main architecture image in a GitHub README.

Create one exceptionally legible editorial architecture diagram, approximately
2400 by 1500 pixels. Warm ivory background, charcoal typography and connectors,
muted teal for protected obligations, terracotta for the edit path. Flat precise
diagram cards, slight paper depth, generous space, no decorative icon clutter.
Title: "From an explicit task to an inspectable result".

Draw this exact data flow. At the top, one "Task contract" card with the small
caption "Semantics · access · budgets · requested and protected behavior".
Branch into three broad columns with large, readable labels:

1. "Read / Diagnose" → "Answer".
2. "Edit" → "Structured proposal" → "Lowering + native precision" →
   "Literal edit + actual candidate". Show a short side entry labelled
   "Literal edit" joining the last node directly. A second edit option,
   "Compatible complete artifact", enters session history directly without
   lowering. Beneath the edit column, a legible annotation:
   "Keep the proposal and actual endpoint distinct".
3. "Create" → "Complete artifact" with caption "No parent".

All three columns enter one full-width "Session history" card. From it, a
single downward arrow enters "Caller-run checks" with the caption
"Structural · behavioral · certificates, where applicable". Then draw
"Select or abstain" → "Seal selection". After sealing, branch into
"Selected candidate" → "Held-out evaluation" → "Receipt" and
"Abstention" → "Receipt". Label held-out evaluation "Caller-owned" and
the abstention branch "No evaluator call". Both branches share the final
receipt node. Abstention must be sealed and receive a receipt too.

Do not connect read, diagnose, or create through edit lowering. Do not depict
recording a candidate as executing checks. Add the readable footer
"Recording a candidate does not run its checks." Do not imply support for
every model architecture, automatic proofs, automatic deployment, or a
universal certificate. All labels must be spelled correctly and remain
readable when the image is displayed about 900 pixels wide. Use deliberate
connector routing, simple arrowheads, no crossings, no Mermaid, no logos,
no fabricated measurements, and no text beyond the specified labels.
```

## 3. Enterprise support-routing explanation

Planned filename: `enterprise-routing.png`

Desired composition: approximately 2400 × 1200 pixels.

Alt text: “A supplied outage feature moves from Ordinary Support to Platform
Operations through one readout-weight edit. Protected billing, security, and
ordinary-support scores are checked. The fixture is synthetic; business savings
and behavior on arbitrary tickets are not established.”

```text
Use case: scientific-educational.
Asset type: a GitHub README explanation connecting a weight edit to a practical
support-routing request.

Create one wide, elegant editorial explanation, approximately 2400 by 1200
pixels. Ivory paper, charcoal text, muted teal for protected behavior and
terracotta for the requested change. Large readable typography and generous
space. Title: "One outage. The right team. Explicit protection.".

Use three clearly separated panels, read left to right:

LEFT, heading "The request": a small ticket card labelled "Telemetry outage".
Under it a muted old route "Ordinary Support" and a new destination
"Platform Operations" connected by one clear change arrow. Caption:
"Change the supplied outage feature's route".

CENTER, heading "The change": an abstract small readout-weight matrix.
Highlight exactly one cell in terracotta. Connect it to a card labelled
"Actual candidate". Include the caption "Inspect the stored weight change".
Do not draw a changed feature extractor or a model retraining loop.

RIGHT, heading "The obligation": three separate teal-outlined rows labelled
"Billing", "Security", and "Ordinary support". Place a shared caption
"Check protected supplied-feature scores". These are check obligations,
not claims about all future tickets. Beneath them a small, distinct card
"If checks fail: stop". Do not show a deployment button or money. The smaller
Distro fixture raises an error if its checks fail; do not depict a session
abstention or a receipt for that failed fixture generation.

Along the bottom put a readable neutral footer:
"Synthetic fixture · fixed supplied features · no production or savings claim".

The image explains the smaller Distro demo fixture: one changed readout weight,
not the full enterprise study. Do not invent ticket counts, accuracy, money,
percentages, customer names, proof seals, arbitrary-text generalization, or
automatic approvals. Use clear connectors only where specified. No Mermaid,
neon, decorative robot or brain icons, interface screenshots, or watermarks.
```

## 4. Distro review architecture

Planned filename: `distro-review.png`

Desired composition: approximately 2400 × 1400 pixels.

Alt text: “Distro's coding host and a workspace-bound companion inspect an exact
change and its evidence through the governance runtime. A separate human
decision CLI controls approval, rejection, and revocation. Independently retained
pins and hashed attachments bind status and complete-history export; eligibility
is limited to a local pilot.”

```text
Use case: infographic-diagram.
Asset type: a GitHub README architecture image for the optional Distro integration.

Create one legible editorial diagram, approximately 2400 by 1400 pixels. Ivory
background, charcoal type and arrows, muted teal for evidence and terracotta
for the human decision boundary. Large labels, restrained linework, generous
spacing. Title: "Evidence in the host. Decisions with the reviewer.".

Use a clear left-to-right central flow:
"Exact change + evidence" → "Distro coding host" →
"Workspace-bound companion" → "Governance runtime" →
"Pinned status + complete-history export" → "Local-pilot eligibility result".
Caption the final card "True only for a correctly bound human approval" and
"Production authority: false · Scientific authority: false". A status check or export
must not appear to grant eligibility by itself.

Group the host and companion with a subtle outline labelled "Optional Distro
integration". Beneath the companion, list the four operations in plain English:
"Inspect · Prepare review · Check status · Export history".

Above the governance runtime, draw a separate card "Human reviewer" and a
smaller card "Independent decision CLI". Connect the human reviewer through
the decision CLI to the runtime with arrows labelled "Approve · Reject · Revoke".
Do not draw any approval arrow from the model, host, or companion.

Beneath status/export, draw an input card "Independently retained pins" with
caption "Original request + current head" and an arrow into status/export.
Add the legible annotation "Rebuild the current request; hash attachment bytes".
Add the short footer "Evidence binding is not evidence verification.".

A separate bottom note, readable at README width, must say:
"Authentication, trusted latest-head storage, concurrency, and operational
enforcement remain external.".

This is a conceptual architecture, not a captured live host session. Do not
depict automatic proof execution, authenticated reviewer identity, production
authorization, a verified Python-to-Lean refinement, or validated live model
tool invocation. Local-pilot eligibility must be the final authority label.
Do not add deployment arrows, proof seals, fake metrics, Mermaid, logos,
cloud-icon filler, neon, or watermarks. All arrow directions and labels must
match this specification.
```

## Placement and acceptance

Place the hero immediately below the README title; keep the project description
as selectable Markdown. Put the enterprise illustration beside the smaller
Distro fixture explanation so it is not mistaken for the larger enterprise study.
Replace the existing architecture embed only after the new image passes review.
Place the Distro architecture with its review workflow.

Inspect each full-resolution asset and its README-width rendering. Verify exact
wording, task branches, human authority, and assumptions against the source
documentation. Preserve accessible Markdown explanations and descriptive alt
text. Record actual asset identities only after generation succeeds; do not
invent hashes, dimensions, or a visual-review result.

Existing diagrams retain their independent
[renderer provenance](../flowchart-manifest.json). The existing geometry
illustration retains its
[source and assumptions](../../../artifacts/explanations/readme-edit/source.json).
Neither provenance record describes the planned ChatGPT artwork.
