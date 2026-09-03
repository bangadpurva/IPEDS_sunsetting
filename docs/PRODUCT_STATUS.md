# Student Pathways product status

## Completed foundation

- Explainable IPEDS/BLS pathway rankings.
- Hard credential constraints and domain-filtered occupation suggestions.
- Rules-first coaching that works without an LLM.
- Multi-turn chat sessions with location and annual-cost extraction.
- Optional College Scorecard institution enrichment.
- Optional O*NET occupation, skill, and technology enrichment.
- Human-readable source freshness dates.
- Health check, chat request limits, bounded session storage, and disabled-by-default administrative endpoints.
- Container deployment configuration.
- A publishable Viascope web experience with explainable field, career, and institution views.
- Side-by-side shortlists for up to three study routes and up to three institutions.
- Explicit loading, partial-data, empty, and retry states in the explorer.

## Next product validation work

- Obtain an O*NET key and generate the first occupation-enrichment cache.
- Validate CIP-SOC recommendations with students and career advisors.
- Validate College Scorecard cost and outcome explanations with prospective learners.
- Validate proximity defaults and distance explanations with prospective learners.
- Decide whether annual-cost preference means sticker price or average net price.
- Add anonymous recommendation feedback and outcome metrics.
- Move chat sessions to a privacy-conscious persistent store only if users need saved plans.
- Conduct accessibility, security, and methodology reviews before a public beta.

The next release criterion should be: a learner can enter interests, credential, location, and budget; receive relevant careers; compare three schools; and understand the source and limitation of every important number. The website now supports this complete functional path, including ZIP/city lookup, browser location, 100-mile and in-state modes, proximity sorting, and straight-line distance labels.

## Current validation baseline

- Python advisor, coach, and external-data suite: 9 passing tests.
- Website quality gates: ESLint and the production vinext build pass.
- Institution comparison: selection is capped at three; selected state is exposed with `aria-pressed`; and Scorecard average net price, tuition, completion, and earnings are shown with explicit limitations.
- College Scorecard cache: targeted by the 300 institutions in the comparison dataset, with 189 current institution matches and cost data on 259 of 300 displayed program records after the UNITID join.
- Location logic: 4 passing frontend tests cover input parsing, real-world distance plausibility, lookup normalization, and invalid-input errors. A live ZIP lookup is also part of release validation.

## Existing website features

- Interest and work-style exploration with explainable field matching.
- Credential, evidence-priority, state, affordability, and proximity controls.
- Career growth, openings, wages, and typical education views.
- College program discovery enriched with Scorecard cost and outcomes.
- Three-route and three-school comparison workspaces.
- ZIP/city lookup, optional browser location, and distance-based ordering.
- Methodology, source periods, limitations, partial-data states, and privacy guidance.
- Responsive, keyboard-accessible public and product surfaces.

## Needed product features

- Bring the existing rules-first coach into the published Viascope experience.
- Add anonymous usefulness feedback and funnel/outcome analytics.
- Add direct official-school and financial-aid links with freshness checks.
- Distinguish online, hybrid, and campus delivery where source data supports it.
- Save, share, or export a comparison without requiring an account.
- Add program-level outcomes where disclosure quality is sufficient.
- Run learner/advisor usability, accessibility, security, and methodology reviews before public beta.
