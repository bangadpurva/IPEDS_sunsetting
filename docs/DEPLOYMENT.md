# Deployment and data operations

The application can run without a paid LLM. Recommendations and coaching remain available through the deterministic IPEDS/BLS advisor.

## Local beta

```bash
python3 app/build_student_data.py
python3 -B -m app.ipeds_connect.server
```

Open `http://127.0.0.1:8000`. Administrative refresh and research execution are disabled by default. Enable them only for a trusted local session:

```bash
ENABLE_ADMIN_ENDPOINTS=1 python3 -B -m app.ipeds_connect.server
```

For a remotely reachable administrative service, also set `ADMIN_TOKEN` and call administrative endpoints with the `X-Admin-Token` header. The browser UI intentionally does not store or submit this token.

## Free data enrichments

Register for free College Scorecard and O*NET keys, then build local caches:

```bash
export COLLEGE_SCORECARD_API_KEY="..."
python3 -B -m app.ipeds_connect.scorecard

export ONET_API_KEY="..."
python3 -B -m app.ipeds_connect.onet --limit 100

python3 app/build_student_data.py
```

The cache files are placed in `data_cache/`. They are optional: an absent or stale external cache never prevents the base IPEDS/BLS application from building. College Scorecard enriches institutions with price and outcomes; O*NET enriches occupations with descriptions, skills, and technology.

## Container

```bash
docker build -t ipeds-student-pathways .
docker run --rm -p 8000:8000 ipeds-student-pathways
```

The image exposes `/healthz`, binds to `0.0.0.0`, and keeps all administrative endpoints disabled. Set `PORT` when required by a hosting platform.

## Production checklist

- Run research and external API synchronization as offline scheduled jobs, not from public requests.
- Serve the generated datasets read-only to the web process.
- Keep administrative endpoints disabled or behind a separately authenticated internal service.
- Terminate TLS at the hosting platform or reverse proxy.
- Add persistent session storage before running multiple API replicas; the beta chat store is intentionally in memory.
- Add centralized logs and error monitoring without recording raw student prompts.
- Publish the methodology, source dates, limitations, privacy statement, and non-guarantee disclaimer.
- Test keyboard navigation, screen readers, mobile layouts, and color contrast before a public launch.

## Current limits

- Chat sessions are stored in memory and disappear when the process restarts.
- Each IP address is limited to 30 chat turns per minute.
- The application does not create user accounts or intentionally collect personal identifiers.
- College Scorecard and O*NET availability is best-effort; cached data prevents runtime dependency on either service.
