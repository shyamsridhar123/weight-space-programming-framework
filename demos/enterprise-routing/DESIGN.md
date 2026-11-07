---
name: "Enterprise Routing Demo"
description: "A quiet change-review sheet for the local Weight Space Programming enterprise demonstration."
colors:
  paper: "#f7f8f5"
  surface: "#ffffff"
  ink: "#1d302d"
  muted: "#52635f"
  rule: "#d9e1dc"
  accent: "#17634f"
  accent-deep: "#104b3d"
  wash: "#edf5ef"
  warning: "#8c4b17"
  warning-wash: "#fff3e6"
  focus: "#2e7c64"
  control-rule: "#98aaa0"
  nav-hover: "#e5eee7"
  nav-current: "#dcece2"
  secondary-hover: "#e4eee7"
  changed-cell: "#d5eadc"
  table-rule: "#b7c8bd"
typography:
  display:
    fontFamily: "Georgia, 'Times New Roman', serif"
    fontSize: "clamp(30px, 3vw, 43px)"
    fontWeight: 400
    lineHeight: 1.16
    letterSpacing: "-.025em"
  headline:
    fontFamily: "'Segoe UI', system-ui, sans-serif"
    fontSize: "24px"
    fontWeight: 650
    lineHeight: 1.3
  title:
    fontFamily: "'Segoe UI', system-ui, sans-serif"
    fontSize: "19px"
    fontWeight: 650
    lineHeight: 1.3
  body:
    fontFamily: "'Segoe UI', system-ui, sans-serif"
    fontSize: "17px"
    fontWeight: 400
    lineHeight: 1.5
  lead:
    fontFamily: "'Segoe UI', system-ui, sans-serif"
    fontSize: "18px"
    fontWeight: 400
    lineHeight: 1.5
  label:
    fontFamily: "'Segoe UI', system-ui, sans-serif"
    fontSize: "14px"
    fontWeight: 650
    lineHeight: 1.5
  walkthrough:
    fontFamily: "'Segoe UI', system-ui, sans-serif"
    fontSize: "15px"
    fontWeight: 400
    lineHeight: 1.35
  table:
    fontFamily: "'Segoe UI', system-ui, sans-serif"
    fontSize: "15px"
    fontWeight: 400
    lineHeight: 1.5
  small:
    fontFamily: "'Segoe UI', system-ui, sans-serif"
    fontSize: "14px"
    fontWeight: 400
    lineHeight: 1.5
  formula:
    fontFamily: "Consolas, monospace"
    fontSize: "16px"
    fontWeight: 400
    lineHeight: 1.5
  receipt:
    fontFamily: "Consolas, monospace"
    fontSize: "12px"
    fontWeight: 400
    lineHeight: 1.6
rounded:
  control: "6px"
  default: "12px"
spacing:
  sm: "8px"
  md: "16px"
  lg: "24px"
  xl: "40px"
components:
  button-primary:
    backgroundColor: "{colors.accent}"
    textColor: "{colors.surface}"
    typography: "{typography.label}"
    rounded: "{rounded.control}"
    padding: "11px 17px"
  button-primary-hover:
    backgroundColor: "{colors.accent-deep}"
  button-secondary:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.accent-deep}"
    typography: "{typography.label}"
    rounded: "{rounded.control}"
    padding: "11px 17px"
  button-secondary-hover:
    backgroundColor: "{colors.secondary-hover}"
  button-text:
    backgroundColor: "transparent"
    textColor: "{colors.muted}"
    typography: "{typography.label}"
    rounded: "{rounded.control}"
    padding: "11px 17px"
  button-text-hover:
    backgroundColor: "{colors.nav-hover}"
    textColor: "{colors.ink}"
  scenario-select:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.ink}"
    rounded: "{rounded.control}"
    padding: "9px 7px"
    width: "100%"
  walkthrough-step:
    backgroundColor: "transparent"
    textColor: "{colors.muted}"
    typography: "{typography.walkthrough}"
    rounded: "{rounded.control}"
    padding: "11px 12px"
  walkthrough-step-current:
    backgroundColor: "{colors.nav-current}"
    textColor: "{colors.accent-deep}"
  policy-card:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.ink}"
    rounded: "{rounded.default}"
    padding: "24px"
  decision:
    backgroundColor: "{colors.wash}"
    textColor: "{colors.ink}"
    rounded: "{rounded.default}"
    padding: "22px 26px"
  decision-warning:
    backgroundColor: "{colors.warning-wash}"
    textColor: "{colors.ink}"
    rounded: "{rounded.default}"
    padding: "22px 26px"
  evidence-table:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.ink}"
    typography: "{typography.table}"
    width: "100%"
  evidence-row-selected:
    backgroundColor: "{colors.wash}"
  matrix-cell-changed:
    backgroundColor: "{colors.changed-cell}"
    textColor: "{colors.accent-deep}"
---

# Design System: Enterprise Routing Demo

## Overview

**Creative North Star: "The Quiet Change-Review Sheet"**

This design system applies to the enterprise routing demo. Its confirmed direction is quiet, light, simple, and readable: paper surrounds a white reading surface, dark green ink anchors the content, and modest green or amber signals make decisions easy to scan. The interface keeps a steady review cadence rather than asking the reader to interpret decorative chrome.

Serif headings introduce the decision; familiar system sans-serif text carries the explanation and controls. Tables, rules, and restrained tonal changes make the evidence inspectable. The rollback receipt is the sole animated result treatment. Detailed page sequencing remains in `direction.md`.

**Key Characteristics:**

- Light paper and flat white surfaces.
- Serif reading headings with compact system controls.
- Green for passing evidence; amber for blocked evidence.
- Bordered tables with numeric alignment and local change emphasis.
- Visible keyboard focus and a responsive walkthrough rail.

## Colors

The palette pairs quiet green neutrals with one affirmative green family and an amber status family. The frontmatter is the normative value source; the names below explain its application.

### Primary

- **Review Green** (`accent`): primary action fills, links, passing values, and the new route.
- **Deep Review Green** (`accent-deep`): primary hover, selected navigation text, secondary action text, and the changed matrix value.
- **Pass Wash** (`wash`): selected candidate rows and successful decision surfaces.
- **Changed Coefficient Wash** (`changed-cell`): the single edited weight cell.
- **Focus Green** (`focus`): the shared keyboard outline on buttons, select controls, links, and disclosures.
- **Rail Selection Wash** (`nav-current`): the current walkthrough step.
- **Rail Hover Wash** (`nav-hover`): navigation and text-button hover.
- **Secondary Hover Wash** (`secondary-hover`): bordered secondary-button hover.

### Secondary

- **Blocked Amber** (`warning`): failed or ineligible evidence values.
- **Blocked Paper** (`warning-wash`): conflict, rejection, abstention, and error surfaces.

### Neutral

- **Light Paper** (`paper`): the page ground.
- **White Surface** (`surface`): masthead, policy containers, tables, fields, and secondary controls; also primary-button text.
- **Green Ink** (`ink`): main text and labels.
- **Quiet Ink** (`muted`): lead text, explanatory notes, inactive navigation, and table headings.
- **Sheet Rule** (`rule`): layout divisions, card borders, row separators, and formula frames.
- **Control Rule** (`control-rule`): select and secondary-button borders.
- **Table Header Rule** (`table-rule`): the slightly stronger table header divider.

**The Evidence Color Rule.** Keep pass, selection, focus, and changed-cell treatments tied to their existing roles. Green and amber carry readable text labels; color alone never conveys a decision.

## Typography

**Display Font:** Georgia with Times New Roman and serif fallbacks.

**Body Font:** Segoe UI with system-ui and sans-serif fallbacks.

**Label/Mono Font:** Controls share the system sans-serif stack; formulas and artifact identities use Consolas with a monospace fallback.

**Character:** Serif headings give the review a calm reading voice. The system sans-serif stays familiar and economical at the denser control and evidence levels. No external font load is required.

### Hierarchy

- **Display:** the `display` token introduces each reading step. It uses balanced wrapping and a maximum width of 28ch. At the narrow breakpoint its size becomes 33px.
- **Headline:** the `headline` token identifies a decision result inside a tonal panel.
- **Title:** the `title` token marks a section or policy obligation.
- **Body:** the `body` token carries ordinary explanation; the narrow layout uses 16px.
- **Lead:** the `lead` token introduces the step in quiet ink, with a maximum width of 68ch; the narrow layout uses 17px.
- **Label and walkthrough:** compact system text identifies controls and steps. Selected steps use weight 650.
- **Table:** tabular numerals align evidence. Standard header labels are 13px at weight 650; matrix headers are 12px.
- **Small:** subdued explanatory notes remain readable. Run status and page footer use the observed 13px variation.
- **Formula and receipt:** monospace isolates coefficient changes and identities. Hashes wrap anywhere rather than extending the page width.

**The Numeric Alignment Rule.** Use tabular numerals and right alignment for comparison metrics. Preserve the matrix's centered cells and its left-aligned queue labels.

## Layout

The desktop workspace has a persistent rail and a fluid main column within a centered maximum width of 1480px. The rail is 250px wide; the main surface has 35px top, 46px horizontal, and 24px bottom padding. A white masthead uses 23px by 40px padding, and rules divide the major regions. The article's desktop minimum height is 488px so the step controls retain a steady position across short screens.

The declared spacing scale is the observed small, medium, large, and extra-large rhythm in the frontmatter. Paired policy surfaces use equal columns with a 30px gap. The impact pairing uses a 1.2fr to 1fr ratio with a 32px gap. Tables fill their available width and scroll horizontally inside their wrapper when their evidence needs more space.

At a maximum width of 1000px, the rail narrows to 220px, main padding becomes 32px by 28px by 24px, paired policy gaps become 20px, and policy padding becomes 18px. The step footer stacks its concept and actions.

At a maximum width of 700px, the workspace becomes one column. The rail becomes a top section with a horizontally scrolling row of steps. Scenario controls become a label spanning a two-column row, followed by the select and run action. Main padding becomes 25px by 20px, paired sections become one column, article minimum height is removed, and footer actions distribute across the width. Navigation focuses the main reading region and brings it into view on this layout.

## Elevation & Depth

The system is flat. It contains no box shadows. White surfaces, green or amber washes, and thin rules distinguish regions and states. Hover changes tone immediately; it does not lift or scale the control.

**The Flat Review Rule.** Use the existing tonal and border hierarchy when extending this demo. Reserve motion for a meaningful result rather than introducing floating surfaces.

## Shapes

Controls, navigation steps, standalone formulas, and error messages have gently curved corners using `rounded.control`. Policy and decision surfaces use `rounded.default`. Tables remain rectangular and share continuous row rules. The policy's inline formula is unframed, with square corners and transparent background. These shapes keep evidence prominent without creating nested decorative containers.

## Components

### Buttons

Compact, familiar actions use the label typography, control radius, 11px by 17px padding, and a minimum height of 43px. Primary actions use Review Green and White Surface; hover deepens the fill. Secondary actions use a white surface, deep-green text, and the Control Rule border; hover adds Secondary Hover Wash. Text actions use quiet ink and transparent fill; hover adds Rail Hover Wash and Green Ink.

All button variants use a 3px Focus Green outline offset by 4px. Disabled buttons use opacity 0.65 and a wait cursor. There is no authored press transform or timed hover transition.

### Scenario Field

The native select fills the rail width, with a white surface, Green Ink, the Control Rule border, control corners, and 9px by 7px padding. Its text is 14px. The label uses the label role; the run status remains subdued and readable. The select receives the shared focus outline, and the application disables it while a run is pending. This demo has no text-input or chip component to extract.

### Walkthrough Navigation

Steps use transparent resting surfaces, quiet text, left alignment, control corners, and 11px by 12px padding. Hover adds Rail Hover Wash. The current step is identified with `aria-current="step"`, Rail Selection Wash, Deep Review Green, and weight 650. The mobile presentation keeps every step reachable through horizontal scrolling.

### Policy Containers

White, thinly bordered containers use the default surface radius and 24px padding, reducing to 18px at the intermediate breakpoint. Section headings start flush with the content. The internal formula appears as plain monospace evidence without an additional card frame.

### Decision Surfaces

Green or amber tonal panels use the default radius, 22px by 26px padding, and a larger result heading. Written decisions identify the result even when color is unavailable. A verified restoration applies `receipt-reveal`: a 650ms clip reveal with `cubic-bezier(.16,1,.3,1)`. Reduced-motion preference disables this animation.

### Evidence Tables

Comparisons use white backgrounds, collapsed borders, thin row separators, and a stronger header rule. Headers stay on one line; the surrounding wrapper scrolls horizontally. Candidate names are actual underlined buttons. The selected candidate row receives Pass Wash and the button exposes `aria-pressed`. Pass and fail cells combine the appropriate status color with weight 650.

### Weight Matrix

The matrix keeps queue labels aligned left and centers the feature values. The edited cell alone receives Changed Coefficient Wash, Deep Review Green, and weight 650. It shows the realized endpoint rather than a decorative matrix illustration.

### Route and Receipt Evidence

The route change uses Georgia at 28px, reducing to 24px on narrow screens, with a small inline arrow and green destination text. Bound evidence is available in native details/summary disclosures. Identity values use the receipt role and wrapping; disclosure controls share the keyboard focus outline.

## Do's and Don'ts

### Do:

- **Do** preserve the light paper, white surface, thin-rule hierarchy.
- **Do** use green and amber with explicit result text.
- **Do** preserve visible focus on every interactive primitive.
- **Do** retain readable small notes, tabular numeric evidence, and wrapping identities.
- **Do** keep table evidence horizontally scrollable and stack paired reading sections at the narrow breakpoint.
- **Do** highlight the actual edited coefficient and the selected candidate locally.

### Don't:

- **Don't** introduce shadows or lifting hover effects into this flat review grammar.
- **Don't** use status color as a decorative fill across unrelated content.
- **Don't** shrink or clip evidence to force dense tables inside a mobile viewport.
- **Don't** give every coefficient the changed-cell treatment.
- **Don't** add motion to routine navigation or disable the reduced-motion treatment.
