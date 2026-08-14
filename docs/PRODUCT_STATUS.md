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

## Next product validation work

- Obtain free API keys and generate the first Scorecard and O*NET caches.
- Validate CIP-SOC recommendations with students and career advisors.
- Add a detailed institution comparison workspace rather than only enriched trend rows.
- Resolve locations to states or coordinates and rank institutions by distance.
- Decide whether annual-cost preference means sticker price or average net price.
- Add anonymous recommendation feedback and outcome metrics.
- Move chat sessions to a privacy-conscious persistent store only if users need saved plans.
- Conduct accessibility, security, and methodology reviews before a public beta.

The next release criterion should be: a student can enter interests, credential, location, and budget; receive relevant careers; compare at least three schools; and understand the source and limitation of every important number.
