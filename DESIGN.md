---
name: Viascope
description: A clear, optimistic decision workspace for lifelong learning.
colors:
  signal-violet: "#5146E5"
  signal-violet-deep: "#3428BD"
  action-lime: "#DFFF58"
  momentum-coral: "#FF705D"
  midnight-ink: "#16172C"
  supporting-text: "#5E6078"
  workspace: "#F7F7FA"
  surface: "#FFFFFF"
  divider: "#DEDEE8"
  positive: "#167F48"
  negative: "#B3483E"
typography:
  display:
    fontFamily: "Geist, Arial, Helvetica, sans-serif"
    fontSize: "clamp(3.4rem, 6vw, 5.8rem)"
    fontWeight: 780
    lineHeight: 0.96
    letterSpacing: "-0.04em"
  headline:
    fontFamily: "Geist, Arial, Helvetica, sans-serif"
    fontSize: "2.1rem"
    fontWeight: 780
    lineHeight: 1.12
    letterSpacing: "-0.035em"
  title:
    fontFamily: "Geist, Arial, Helvetica, sans-serif"
    fontSize: "1.05rem"
    fontWeight: 750
    lineHeight: 1.3
  body:
    fontFamily: "Geist, Arial, Helvetica, sans-serif"
    fontSize: "1rem"
    fontWeight: 400
    lineHeight: 1.62
  label:
    fontFamily: "Geist, Arial, Helvetica, sans-serif"
    fontSize: "0.75rem"
    fontWeight: 750
    lineHeight: 1.35
rounded:
  control: "9px"
  container: "12px"
  panel: "14px"
  pill: "999px"
spacing:
  xs: "5px"
  sm: "8px"
  md: "12px"
  lg: "16px"
  xl: "24px"
  xxl: "32px"
components:
  button-primary:
    backgroundColor: "{colors.action-lime}"
    textColor: "{colors.midnight-ink}"
    typography: "{typography.label}"
    rounded: "{rounded.control}"
    padding: "12px 18px"
  button-secondary:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.signal-violet}"
    typography: "{typography.label}"
    rounded: "{rounded.control}"
    padding: "9px 12px"
  input:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.midnight-ink}"
    typography: "{typography.body}"
    rounded: "{rounded.control}"
    padding: "11px 12px"
  choice-selected:
    backgroundColor: "#EEECFF"
    textColor: "{colors.signal-violet-deep}"
    typography: "{typography.label}"
    rounded: "{rounded.control}"
    padding: "9px 10px"
  data-container:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.midnight-ink}"
    rounded: "{rounded.container}"
    padding: "16px"
---

# Design System: Viascope

## 1. Overview

**Creative North Star: "The Open Map"**

Viascope should feel like unfolding a clear map on a table with a capable guide nearby. It presents many routes without making the landscape intimidating: open white space, firm structure, bright navigational signals, and dense evidence revealed exactly when it becomes useful. The system is optimistic and current, but every expressive choice must improve orientation or confidence.

The public surface may use larger compositions and the full palette. The product surface is more restrained: neutral workspace layers, predictable controls, compact tables, and Signal Violet reserved for current position and primary state. Tactility comes from crisp strokes and small, directional structural shadows—not ambient decoration.

The system explicitly rejects legacy university portals, dense government-data dashboards, high-pressure admissions funnels, opaque personality quizzes, generic AI-product aesthetics, and inaccessible color-only meaning.

**Key Characteristics:**

- Bright navigation signals on a calm neutral workspace
- Tactile, confident controls with crisp edges
- Progressive disclosure inspired by Airbnb's approachable hierarchy
- Dense evidence organized for scanning rather than decoration
- Plain language for learners at every stage of life

## 2. Colors

The palette behaves like map notation: violet locates the user, lime invites movement, coral calls out momentum, and dark ink keeps the system trustworthy.

### Primary

- **Signal Violet:** The brand anchor, active navigation state, focus treatment, and current selection. Never use it as decorative wash across an entire product screen.
- **Signal Violet Deep:** Hover, pressed, and high-contrast violet text.

### Secondary

- **Action Lime:** Primary calls to action, selected priority context, and directional highlights. Always pair it with Midnight Ink.

### Tertiary

- **Momentum Coral:** A limited emphasis for change, opportunity, and featured map routes. It is not an error color.
- **Positive:** Confirmed favorable numeric direction with a non-color label or symbol.
- **Negative:** Confirmed unfavorable numeric direction with a non-color label or symbol.

### Neutral

- **Midnight Ink:** Primary text, high-confidence outlines, and dark navigation surfaces.
- **Supporting Text:** Explanations, metadata, and secondary labels that still meet contrast requirements.
- **Workspace:** The application canvas and secondary toolbar layer.
- **Surface:** Primary reading and interaction surfaces.
- **Divider:** Table rules, panel separation, and inactive borders.

**The Map-Key Rule.** Signal Violet means location or selection; Action Lime means movement or action; Momentum Coral means noteworthy change. Never swap these roles casually.

**The No Color Alone Rule.** Every status color requires a word, value, icon, or shape that communicates the same meaning.

## 3. Typography

**Display Font:** Geist with Arial and Helvetica fallbacks  
**Body Font:** Geist with Arial and Helvetica fallbacks  

**Character:** A single contemporary sans keeps the application familiar and fast. Weight, scale, and spacing create hierarchy without introducing a decorative voice that competes with the data.

### Hierarchy

- **Display:** Heavy and tightly composed for the public landing hero only; balance lines and never exceed the documented tracking floor.
- **Headline:** Compact product-page titles and major section headings.
- **Title:** Panel, route, table, and comparison titles.
- **Body:** Explanations and guidance, capped at 70 characters per line when rendered as prose.
- **Label:** Buttons, field labels, tabs, metadata, and compact table headers. Sentence case is the default.

**The One-Family Rule.** Product UI always uses the Geist family. Do not introduce a display face into controls, tables, or onboarding.

**The Plain-Language Rule.** A learner must understand a label before they need to understand the underlying dataset.

## 4. Elevation

Viascope is flat by default. Depth is conveyed through tonal layering and borders inside the product. Small, hard-edged structural shadows are reserved for high-value tactile moments such as the landing search control, possibility map, and selected comparison—not repeated across every container.

### Shadow Vocabulary

- **Control press:** A compact dark offset shadow that makes one primary action feel physical.
- **Brand lift:** A violet offset shadow used only on the public possibility map.
- **Comparison lift:** A lime offset shadow reserved for shortlisted options.

**The Earned Lift Rule.** A shadow must indicate action, selection, or a signature brand moment. Static data tables and routine panels remain flat.

## 5. Components

Components feel tactile and confident while retaining standard web affordances.

### Buttons

- **Shape:** Gently squared control corners.
- **Primary:** Action Lime with Midnight Ink, strong label weight, and generous touch target.
- **Hover / Focus:** Hover deepens the lime; focus uses a clearly visible Signal Violet ring; active removes some visual offset to suggest pressure.
- **Secondary:** White or transparent with a crisp divider or violet outline. Destructive actions use text and the Negative role together.

### Chips

- **Style:** Quiet neutral background when inactive; pale violet with violet text and stroke when selected.
- **State:** Include a check or radio mark in addition to the selected color. Pills are reserved for small statuses, never full-size actions.

### Cards / Containers

- **Corner Style:** Controlled medium rounding for panels and data containers.
- **Background:** White against the Workspace canvas; tinted lavender or lime only for meaningful state.
- **Shadow Strategy:** Flat unless the container is a selected comparison or signature brand map.
- **Border:** Crisp Divider strokes or a strong Midnight Ink outline for tactile signature moments.
- **Internal Padding:** Compact in data tables and comfortable in onboarding or explanatory panels.

### Inputs / Fields

- **Style:** White background, clear neutral stroke, readable placeholder, and standard control height.
- **Focus:** Signal Violet border with an outer pale-violet ring.
- **Error / Disabled:** Pair state color with explicit text; disabled controls retain legible content and remove action affordance.

### Navigation

Navigation is stable, compact, and familiar. The wordmark anchors the upper left, current sections use text plus violet state, and mobile views remove secondary links before compressing labels.

### Possibility Map

The signature map connects a user's starting point to study and career signals. It may use the full palette, directional lines, and one Brand Lift shadow. Keep labels concrete and never suggest that a route is predetermined.

### Evidence Tables

Tables prioritize scanning: descriptive first column, consistently aligned numbers, plain-language headers, explicit positive/negative text, and progressive column removal on smaller screens. Never squeeze all desktop columns onto mobile.

## 6. Do's and Don'ts

### Do:

- **Do** use Signal Violet for current position, focus, and selection.
- **Do** use Action Lime for a single primary movement on a surface.
- **Do** reveal complexity progressively, following Airbnb's approachable entry and clear hierarchy.
- **Do** keep every essential interaction keyboard accessible with visible focus.
- **Do** pair data with source, period, meaning, and uncertainty.
- **Do** adapt density structurally on mobile by prioritizing columns and stacking controls.

### Don't:

- **Don't** resemble a legacy university portal or a dense government-data dashboard.
- **Don't** build a high-pressure admissions funnel or an opaque personality quiz that presents an answer as destiny.
- **Don't** use generic AI-product aesthetics, decorative dashboards, or repeated ghost cards.
- **Don't** infantilize adult learners or assume age, income, family support, or prior education.
- **Don't** bury uncertainty or imply guaranteed education, employment, wage, or admissions outcomes.
- **Don't** require research terminology before a learner receives value.
- **Don't** communicate status through color alone.
- **Don't** add soft ambient shadows to routine panels or nest cards inside cards.
