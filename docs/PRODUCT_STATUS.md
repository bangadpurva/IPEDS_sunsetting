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
- Resolve locations to states or coordinates and rank institutions by distance.
- Decide whether annual-cost preference means sticker price or average net price.
- Add anonymous recommendation feedback and outcome metrics.
- Move chat sessions to a privacy-conscious persistent store only if users need saved plans.
- Conduct accessibility, security, and methodology reviews before a public beta.

The next release criterion should be: a learner can enter interests, credential, location, and budget; receive relevant careers; compare three schools; and understand the source and limitation of every important number. The website now supports interests, credentials, state and average-net-price filtering, careers, and three-school comparison. Location-distance ranking remains outstanding.

## Current validation baseline

- Python advisor, coach, and external-data suite: 9 passing tests.
- Website quality gates: ESLint and the production vinext build pass.
- Institution comparison: selection is capped at three; selected state is exposed with `aria-pressed`; and Scorecard average net price, tuition, completion, and earnings are shown with explicit limitations.
- College Scorecard cache: targeted by the 300 institutions in the comparison dataset, with 189 current institution matches and cost data on 259 of 300 displayed program records after the UNITID join.
