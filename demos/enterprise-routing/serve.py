"""Serve a local, synthetic walkthrough backed by the actual framework example."""
from __future__ import annotations

import argparse
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples.enterprise_support_routing import (  # noqa: E402
    SCENARIOS, baseline, run_scenario, scores, routes,
)
from rlcd.weight_space import TensorArtifact, apply_program  # noqa: E402
from rlcd.weight_space.program import Program  # noqa: E402


def run_demo(scenario: str) -> dict:
    if scenario not in SCENARIOS:
        raise ValueError("Choose policy-change, conflict, or coverage-gap")
    return run_scenario(scenario)


def restore_model(scenario: str) -> dict:
    """Apply the published inverse to its exact candidate; return actual scores."""
    bundle = run_demo(scenario)
    if bundle["final_admission"]["decision"] != "admitted_for_synthetic_trial":
        raise ValueError("Only an admitted synthetic trial can be restored in this demo")
    selected = next(row for row in bundle["candidates"]
                    if row["label"] == bundle["final_admission"]["selected_label"])
    candidate = TensorArtifact.from_dict(selected["candidate"]["artifact"])
    restored = apply_program(candidate, Program.from_dict(bundle["rollback"]["program"]))
    if restored.sha256 != baseline().sha256:
        raise RuntimeError("Restored artifact does not match the exact parent")
    queues, margins = routes(scores(restored, bundle["inputs"]["heldout"]), .5)
    return {"restored_parent_sha256": restored.sha256,
            "parent_sha256": bundle["final_admission"]["parent_sha256"],
            "exact_parent_restored": True,
            "tickets": [{"ticket_id": ticket["ticket_id"], "queue": queue,
                         "margin": float(margin)}
                        for ticket, queue, margin in zip(bundle["inputs"]["heldout"], queues, margins)],
            "scope": bundle["rollback"]["scope"]}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(Path(__file__).parent), **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()

    def do_GET(self):
        if self.path.split("?", 1)[0] not in ("/", "/index.html", "/style.css", "/app.js"):
            self.send_error(404)
            return
        super().do_GET()

    def do_HEAD(self):
        if self.path.split("?", 1)[0] not in ("/", "/index.html", "/style.css", "/app.js"):
            self.send_error(404)
            return
        super().do_HEAD()

    def do_POST(self):
        if self.path not in ("/api/run", "/api/rollback"):
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 1024:
                raise ValueError("A small JSON request is required")
            request = json.loads(self.rfile.read(length))
            if not isinstance(request, dict) or set(request) != {"scenario"}:
                raise ValueError("Request must contain only scenario")
            result = (run_demo if self.path == "/api/run" else restore_model)(request["scenario"])
            payload, status = {"ok": True, "result": result}, 200
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            payload, status = {"ok": False, "error": str(exc)}, 400
        except Exception:
            # Local error details belong in the server log, never in the recording.
            import traceback
            traceback.print_exc()
            payload, status = {"ok": False, "error": "Framework run failed. Check the local server log and run again."}, 500
        data = (json.dumps(payload, allow_nan=False) + "\n").encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8876)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Enterprise routing demo: http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
