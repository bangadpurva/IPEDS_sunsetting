from __future__ import annotations

import json
import os
import hmac
import threading
import time
import uuid
from collections import defaultdict, deque
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .advisor import StudentProfile, agentic_recommend, recommend
from .data_adapter import ROOT, export_json
from .dimensions import export_dimensions_json
from .llm_coach import coach_with_optional_llm
from .research_engine import research_config, stream_research_pipeline

WEB_ROOT = ROOT / "web"
DATA_PATH = WEB_ROOT / "data" / "programs.json"
DIMENSIONS_PATH = WEB_ROOT / "data" / "dimensions.json"
JOB_LOCK = threading.Lock()
RUN_JOB = {
    "running": False,
    "status": "idle",
    "stage": "Idle",
    "message": "Research pipeline has not started.",
    "started_at": None,
    "finished_at": None,
    "return_code": None,
    "logs": [],
}
CHAT_LOCK = threading.Lock()
CHAT_SESSIONS: dict[str, list[str]] = {}
CHAT_REQUESTS: dict[str, deque[float]] = defaultdict(deque)
MAX_CHAT_SESSIONS = 1000


def _admin_enabled() -> bool:
    return os.getenv("ENABLE_ADMIN_ENDPOINTS", "0").lower() in {"1", "true", "yes"}


def match_institutions(institutions: list[dict], profile: dict, limit: int = 8) -> list[dict]:
    """Apply location and net-price constraints to cached institution facts."""
    location = str(profile.get("location") or "").lower().strip()
    max_cost = profile.get("max_annual_cost")
    matches = []
    for institution in institutions:
        haystack = " ".join(
            str(institution.get(key) or "")
            for key in ("institution_name", "scorecard_name", "city", "scorecard_city", "state", "scorecard_state")
        ).lower()
        if location and location not in haystack:
            continue
        net_price = institution.get("average_net_price")
        if max_cost is not None and (net_price is None or float(net_price) > float(max_cost)):
            continue
        matches.append(dict(institution))
    matches.sort(
        key=lambda row: (
            row.get("completion_rate") is not None,
            float(row.get("completion_rate") or 0),
            float(row.get("median_earnings_10yr") or 0),
        ),
        reverse=True,
    )
    return matches[:limit]


def _set_job(**updates):
    with JOB_LOCK:
        RUN_JOB.update(updates)


def _append_log(line: str):
    if not line:
        return
    with JOB_LOCK:
        RUN_JOB["logs"].append(line)
        RUN_JOB["logs"] = RUN_JOB["logs"][-80:]
        RUN_JOB["message"] = line
        if line.startswith("[LOAD]"):
            RUN_JOB["stage"] = "Loading IPEDS completions"
        elif line.startswith("[COMBINE]"):
            RUN_JOB["stage"] = "Combining program records"
        elif line.startswith("[Threshold Model]") or line.startswith("[LABELS]"):
            RUN_JOB["stage"] = "Applying sunset-risk model"
        elif line.startswith("[BLS]"):
            RUN_JOB["stage"] = "Running BLS alignment analysis"
        elif line.startswith("[SAVE]"):
            RUN_JOB["stage"] = "Saving research outputs"


def _job_snapshot():
    with JOB_LOCK:
        return dict(RUN_JOB, logs=list(RUN_JOB["logs"]))


def _run_research_job():
    _set_job(
        running=True,
        status="running",
        stage="Starting research pipeline",
        message="Launching ipeds_bls_projections.py",
        started_at=time.time(),
        finished_at=None,
        return_code=None,
        logs=[],
    )
    try:
        for line in stream_research_pipeline():
            _append_log(line)
        _set_job(stage="Building student-facing dataset", message="Regenerating web/data/programs.json")
        path = export_json(DATA_PATH, refresh_research=False)
        _append_log(f"[APP] Student dataset rebuilt: {path.relative_to(ROOT)}")
        dimensions_path = export_dimensions_json()
        _append_log(f"[APP] Dimensions dataset rebuilt: {dimensions_path.relative_to(ROOT)}")
        _set_job(
            running=False,
            status="completed",
            stage="Completed",
            message="Research pipeline completed and student data was refreshed.",
            finished_at=time.time(),
            return_code=0,
        )
    except Exception as exc:
        _append_log(f"[ERROR] {exc}")
        _set_job(
            running=False,
            status="failed",
            stage="Failed",
            message=str(exc),
            finished_at=time.time(),
            return_code=1,
        )


class StudentPathwayHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/healthz":
            self._json({"ok": True, "data_ready": DATA_PATH.exists()})
            return
        if parsed.path == "/api/capabilities":
            self._json({"chat": True, "admin": _admin_enabled() and not os.getenv("ADMIN_TOKEN")})
            return
        if parsed.path == "/api/recommend":
            self._recommend(parsed.query)
            return
        if parsed.path == "/api/agent":
            self._agent(parsed.query)
            return
        if parsed.path == "/api/refresh":
            if not self._admin_allowed():
                return
            params = parse_qs(parsed.query)
            run_pipeline = params.get("run", ["0"])[0] in {"1", "true", "yes"}
            try:
                path = export_json(DATA_PATH, refresh_research=run_pipeline)
                dimensions_path = export_dimensions_json()
                self._json(
                    {
                        "ok": True,
                        "path": str(path.relative_to(ROOT)),
                        "dimensions_path": str(dimensions_path.relative_to(ROOT)),
                        "research_pipeline_ran": run_pipeline,
                    }
                )
            except Exception as exc:
                self._json({"ok": False, "error": str(exc), "research_pipeline_ran": run_pipeline}, status=500)
            return
        if parsed.path == "/api/run-research":
            if not self._admin_allowed():
                return
            snapshot = _job_snapshot()
            if snapshot["running"]:
                self._json({"ok": True, "already_running": True, "job": snapshot})
                return
            thread = threading.Thread(target=_run_research_job, daemon=True)
            thread.start()
            self._json({"ok": True, "already_running": False, "job": _job_snapshot()})
            return
        if parsed.path == "/api/run-status":
            self._json({"ok": True, "job": _job_snapshot()})
            return
        if parsed.path == "/api/research":
            self._json(research_config())
            return
        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/chat":
            self._chat()
            return
        self._json({"ok": False, "error": "Not found"}, status=404)

    def _recommend(self, query: str):
        if not DATA_PATH.exists():
            export_json(DATA_PATH)
        dataset = json.loads(DATA_PATH.read_text(encoding="utf-8"))
        params = parse_qs(query)
        profile = StudentProfile(
            interests=tuple(params.get("interests", [""])[0].split(",")),
            degree_level=params.get("degree", [None])[0] or None,
            risk_tolerance=params.get("risk", ["balanced"])[0],
            career_priority=params.get("priority", ["balanced"])[0],
        )
        self._json({"recommendations": recommend(dataset["programs"], profile), "summary": dataset.get("summary", {})})

    def _agent(self, query: str):
        if not DATA_PATH.exists():
            export_json(DATA_PATH)
        dataset = json.loads(DATA_PATH.read_text(encoding="utf-8"))
        params = parse_qs(query)
        prompt = params.get("prompt", [""])[0]
        self._json(coach_with_optional_llm(dataset["programs"], prompt))

    def _chat(self):
        if not self._chat_rate_allowed():
            self._json({"ok": False, "error": "Too many chat requests; try again shortly."}, status=429)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 20_000:
                self._json({"ok": False, "error": "Invalid request size"}, status=400)
                return
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, json.JSONDecodeError):
            self._json({"ok": False, "error": "Invalid JSON"}, status=400)
            return
        prompt = str(payload.get("prompt", "")).strip()
        if not prompt:
            self._json({"ok": False, "error": "Prompt is required"}, status=400)
            return
        session_id = str(payload.get("session_id") or uuid.uuid4().hex)
        with CHAT_LOCK:
            if session_id not in CHAT_SESSIONS and len(CHAT_SESSIONS) >= MAX_CHAT_SESSIONS:
                CHAT_SESSIONS.pop(next(iter(CHAT_SESSIONS)))
            history = CHAT_SESSIONS.setdefault(session_id, [])
            history.append(prompt[:4000])
            del history[:-8]
            combined_prompt = "\n".join(history)
        if not DATA_PATH.exists():
            export_json(DATA_PATH)
        dataset = json.loads(DATA_PATH.read_text(encoding="utf-8"))
        result = coach_with_optional_llm(dataset["programs"], combined_prompt)
        if DIMENSIONS_PATH.exists():
            dimensions = json.loads(DIMENSIONS_PATH.read_text(encoding="utf-8"))
            result["institutions"] = match_institutions(dimensions.get("institutions", []), result["profile"])
        result.update({"ok": True, "session_id": session_id, "turn": len(history)})
        self._json(result)

    def _chat_rate_allowed(self) -> bool:
        now = time.time()
        key = self.client_address[0]
        with CHAT_LOCK:
            requests = CHAT_REQUESTS[key]
            while requests and requests[0] < now - 60:
                requests.popleft()
            if len(requests) >= 30:
                return False
            requests.append(now)
            return True

    def _admin_allowed(self) -> bool:
        if not _admin_enabled():
            self._json({"ok": False, "error": "Administrative endpoints are disabled."}, status=403)
            return False
        expected = os.getenv("ADMIN_TOKEN")
        if expected and not hmac.compare_digest(self.headers.get("X-Admin-Token", ""), expected):
            self._json({"ok": False, "error": "Administrative token required."}, status=403)
            return False
        return True

    def _json(self, payload: dict, status: int = 200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.end_headers()
        self.wfile.write(body)


def main():
    if not DATA_PATH.exists():
        export_json(DATA_PATH)
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    server = ThreadingHTTPServer((host, port), StudentPathwayHandler)
    print(f"Serving IPEDS Student Pathways at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
