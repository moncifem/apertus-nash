"""HTTP front of the reward service (stdlib only), meant to run in its own container.

  POST /reward         {"groups": [job, ...]}   job = {problem, gold, domain, cands: [{thinking, reply, truncated, ...}]}
                       -> {"version", "results": [null | {"rewards": [...], "parts": [...]}], "usage": {cost_usd, calls, ...}, "stats"}
  GET  /reward/health  -> {"ok", "version", "key_ok", "load_error", "reward_min", "reward_max"}
  GET  /reward/stats   -> cumulative usage, cost and counters

Hot update: reward.py, reward_prompt.md and reward_config.json are re-read on every request (reward.py is re-executed when its mtime
changes), so editing them in the mounted directory changes the running reward without restarting the container or the trainer.
`version` identifies the exact code + prompt + config that produced a result.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import threading
import time
import traceback
import types
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(os.environ.get("AGENT_DIR", Path(__file__).resolve().parent))
KEY_PATH = os.environ.get("DEEPSEEK_KEY_PATH", "/run/secrets/deepseek_key")
log = logging.getLogger("reward-service")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

sys.path.insert(0, str(HERE))                     # reward.py imports its sibling module structured_check
_lock = threading.Lock()
_rmod = {"mtime": None, "module": None}
RTOTAL = {"usage": {}, "stats": {}, "groups": 0, "failed": 0}


def load_reward():
    """reward.py, re-executed when the file changed. A broken edit keeps the last good module and reports the error."""
    path = HERE / "reward.py"
    m = path.stat().st_mtime_ns
    with _lock:
        if _rmod["module"] is None or m != _rmod["mtime"]:
            try:
                # compile from the source text: importlib would reuse a stale .pyc for a same-size edit made within the same second
                mod = types.ModuleType("reward_hot")
                mod.__file__ = str(path)
                exec(compile(path.read_text(), str(path), "exec"), mod.__dict__)
                _rmod.update(mtime=m, module=mod, error=None)
                log.info("reward.py (re)loaded")
            except Exception:  # noqa: BLE001
                _rmod["error"] = traceback.format_exc()[-600:]
                _rmod["mtime"] = m
                log.error("reward.py failed to load, keeping the previous version:\n%s", _rmod["error"])
                if _rmod["module"] is None:
                    raise
        return _rmod["module"]


def reward_settings():
    cfg = json.loads((HERE / "reward_config.json").read_text())
    prompt = (HERE / "reward_prompt.md").read_text()
    try:
        key = Path(KEY_PATH).read_text().strip()
    except OSError:
        key = ""                                  # /reward/health still answers and reports key_ok=false; /reward refuses
    ver = hashlib.sha256((HERE / "reward.py").read_bytes() + prompt.encode() + json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
    return cfg, prompt, key, ver


def run_reward(groups: list[dict]) -> dict:
    mod = load_reward()
    cfg, prompt, key, ver = reward_settings()
    if not key:
        raise RuntimeError(f"no DeepSeek key readable at {KEY_PATH}")
    usage: dict = {}
    stats: dict = {}

    def one(job):
        try:
            return mod.score_group(cfg, key, prompt, job, usage, stats)
        except Exception:  # noqa: BLE001
            log.error("reward group failed:\n%s", traceback.format_exc()[-800:])
            stats["exceptions"] = stats.get("exceptions", 0) + 1
            return None

    t0 = time.time()
    with ThreadPoolExecutor(max(1, int(cfg["concurrency"]))) as ex:
        results = list(ex.map(one, groups))
    cost = mod.cost_usd(cfg, usage)
    with _lock:
        for k, v in usage.items():
            if isinstance(v, (int, float)):
                RTOTAL["usage"][k] = RTOTAL["usage"].get(k, 0) + v
        for k, v in stats.items():
            RTOTAL["stats"][k] = RTOTAL["stats"].get(k, 0) + v
        RTOTAL["groups"] += len(groups)
        RTOTAL["failed"] += sum(r is None for r in results)
    return {"version": ver, "results": results, "usage": {**usage, "cost_usd": round(cost, 6), "seconds": round(time.time() - t0, 1)}, "stats": stats}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        try:
            if self.path == "/reward/health":
                load_reward()
                st = reward_settings()
                return self._send(200, {"ok": True, "version": st[3], "key_ok": bool(st[2]), "load_error": _rmod.get("error"),
                                        "reward_min": st[0]["reward_min"], "reward_max": st[0]["reward_max"]})
            if self.path == "/reward/stats":
                cfg, _, _, ver = reward_settings()
                with _lock:
                    return self._send(200, {"version": ver, **RTOTAL, "cost_usd": round(load_reward().cost_usd(cfg, RTOTAL["usage"]), 6)})
        except Exception:  # noqa: BLE001
            return self._send(500, {"error": traceback.format_exc()[-400:]})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/reward":
                return self._send(200, run_reward(body.get("groups") or []))
        except Exception:  # noqa: BLE001
            return self._send(500, {"error": traceback.format_exc()[-400:]})
        self._send(404, {"error": "not found"})


if __name__ == "__main__":
    load_reward()
    ThreadingHTTPServer(("0.0.0.0", int(os.environ.get("PORT", "8090"))), H).serve_forever()
