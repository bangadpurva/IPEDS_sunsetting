from __future__ import annotations

import json
import threading
import time
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
        if parsed.path == "/api/recommend":
            self._recommend(parsed.query)
            return
        if parsed.path == "/api/agent":
            self._agent(parsed.query)
            return
        if parsed.path == "/api/refresh":
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

    def _json(self, payload: dict, status: int = 200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    if not DATA_PATH.exists():
        export_json(DATA_PATH)
    server = ThreadingHTTPServer(("127.0.0.1", 8000), StudentPathwayHandler)
    print("Serving IPEDS Student Pathways at http://127.0.0.1:8000")
    server.serve_forever()


if __name__ == "__main__":
    main()
