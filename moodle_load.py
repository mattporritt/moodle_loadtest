#!/usr/bin/env python3
"""
Moodle LMS load generator with human-readable table output and CSV capture.
- Async requests + login handling
- Progress & final summary printed as tables
- Progress and summary also written to CSV in --stats-dir
"""
import argparse
import asyncio
import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from aiohttp import ClientSession, TCPConnector

def _format_table(headers, rows):
    cols = len(headers)
    widths = [len(h) for h in headers]
    for row in rows:
        for i in range(cols):
            widths[i] = max(widths[i], len(str(row[i]) if i < len(row) else ""))
    sep = "+".join("-" * (w + 2) for w in widths)
    lines = []
    header_line = " | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers))
    lines.append(header_line)
    lines.append(sep)
    for row in rows:
        line = " | ".join(str(row[i]).ljust(widths[i]) for i in range(cols))
        lines.append(line)
    return "\n".join(lines)

LOGIN_TOKEN_RE = re.compile(r'name="logintoken"\s+value="([^"]+)"')

@dataclass
class UserCred:
    username: str
    password: str

@dataclass
class UrlTemplate:
    path: str

@dataclass
class ParamRange:
    min: int
    max: int

@dataclass
class Config:
    base_url: str
    login_path: str
    users: List[UserCred]
    urls: List[UrlTemplate]
    parameters: Dict[str, ParamRange]

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Config":
        users = [UserCred(**u) for u in d["users"]]
        urls = [UrlTemplate(**u) for u in d["urls"]]
        parameters = {k: ParamRange(**v) for k, v in d.get("parameters", {}).items()}
        return Config(
            base_url=d["base_url"].rstrip("/"),
            login_path=d.get("login_path", "/login/index.php"),
            users=users,
            urls=urls,
            parameters=parameters,
        )

@dataclass
class Stats:
    started_at: float = field(default_factory=time.monotonic)
    total_requests: int = 0
    success: int = 0
    failures: int = 0
    latencies: List[float] = field(default_factory=list)
    target_rpm: int = 0

    def record(self, ok: bool, latency: float):
        self.total_requests += 1
        if ok:
            self.success += 1
            self.latencies.append(latency)
        else:
            self.failures += 1

    def snapshot(self) -> Dict[str, Any]:
        elapsed = max(0.0001, time.monotonic() - self.started_at)
        rpm = self.total_requests / (elapsed / 60.0)
        p50, p95, p99 = self._percentiles([50, 95, 99])
        return {
            "elapsed_sec": round(elapsed, 1),
            "target_rpm": self.target_rpm,
            "observed_rpm": round(rpm, 1),
            "total": self.total_requests,
            "success": self.success,
            "failures": self.failures,
            "latency_ms_p50": p50,
            "latency_ms_p95": p95,
            "latency_ms_p99": p99,
        }

    def _percentiles(self, ps: List[int]) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        if not self.latencies:
            return (None, None, None)
        xs = sorted(self.latencies)
        out = []
        for p in ps:
            k = max(0, min(len(xs) - 1, int(round((p / 100.0) * (len(xs) - 1)))))
            out.append(int(xs[k] * 1000))
        return tuple(out)  # type: ignore

def render_path(tpl: str, parameters: Dict[str, ParamRange]) -> str:
    def repl(m):
        name = m.group(1)
        if name not in parameters:
            raise ValueError(f"No parameter range defined for {{{name}}} in URL '{tpl}'")
        pr = parameters[name]
        return str(random.randint(pr.min, pr.max))
    return re.sub(r"\{([a-zA-Z0-9_]+)\}", repl, tpl)

async def fetch_logintoken(session: ClientSession, login_url: str) -> Optional[str]:
    async with session.get(login_url, allow_redirects=True) as resp:
        html = await resp.text()
        m = LOGIN_TOKEN_RE.search(html)
        return m.group(1) if m else None

async def login_user(base_url: str, login_path: str, cred: UserCred, insecure_tls: bool, timeout: int = 20) -> Optional[ClientSession]:
    connector = TCPConnector(ssl=False) if insecure_tls else TCPConnector(ssl=None)
    session = ClientSession(
        connector=connector,
        timeout=aiohttp.ClientTimeout(total=timeout),
        headers={"User-Agent": "moodle-loadgen/1.1"},
        cookie_jar=aiohttp.CookieJar(unsafe=insecure_tls),
    )
    try:
        login_url = f"{base_url}{login_path}"
        token = await fetch_logintoken(session, login_url)
        if not token:
            print(f"[WARN] No logintoken for {cred.username}; attempting login anyway.")

        payload = {"username": cred.username, "password": cred.password, "anchor": ""}
        if token:
            payload["logintoken"] = token

        async with session.post(login_url, data=payload, allow_redirects=True) as resp:
            ok_cookie = any(c.key.lower().startswith("moodlesession") for c in session.cookie_jar)
            ok_status = resp.status in (200, 302, 303)
            if ok_cookie and ok_status:
                return session
            else:
                text = await resp.text()
                if "loginerrormessage" in text or "Invalid login" in text:
                    print(f"[ERROR] Login failed for {cred.username}: invalid credentials.")
                else:
                    print(f"[ERROR] Login maybe failed for {cred.username}: status {resp.status}")
                await session.close()
                return None
    except Exception as e:
        print(f"[ERROR] Login exception for {cred.username}: {e}")
        await session.close()
        return None

async def worker(name: str, base_url: str, job_q: "asyncio.Queue[Tuple[ClientSession, UrlTemplate, Dict[str, ParamRange]]]", stats: Stats):
    while True:
        try:
            session, url_tpl, param_ranges = await job_q.get()
            if session is None:
                job_q.task_done()
                return
            path = render_path(url_tpl.path, param_ranges)
            full_url = f"{base_url}{path}"
            t0 = time.monotonic()
            ok = False
            try:
                async with session.get(full_url, allow_redirects=True) as resp:
                    ok = 200 <= resp.status < 400
                    await resp.read()
            except Exception as e:
                ok = False
                print(f"[{name}] Request error: {e} :: {full_url}")
            dt = time.monotonic() - t0
            stats.record(ok, dt)
            job_q.task_done()
        except asyncio.CancelledError:
            return
        except Exception as e:
            print(f"[{name}] Unexpected worker error: {e}")

async def producer(job_q: "asyncio.Queue[Tuple[ClientSession, UrlTemplate, Dict[str, ParamRange]]]", sessions: List[ClientSession], url_templates: List[UrlTemplate], param_ranges: Dict[str, ParamRange], rpm: int, duration_sec: int):
    interval = 60.0 / max(1, rpm)
    end = time.monotonic() + duration_sec
    while time.monotonic() < end:
        sess = random.choice(sessions)
        tpl = random.choice(url_templates)
        await job_q.put((sess, tpl, param_ranges))
        await asyncio.sleep(interval)

def _csv_header():
    return ["timestamp","elapsed_sec","target_rpm","observed_rpm","total","success","failures","latency_ms_p50","latency_ms_p95","latency_ms_p99"]

def _csv_row(now_iso: str, snap: Dict[str, Any]):
    return [
        now_iso,
        snap["elapsed_sec"],
        snap["target_rpm"],
        snap["observed_rpm"],
        snap["total"],
        snap["success"],
        snap["failures"],
        snap["latency_ms_p50"] if snap["latency_ms_p50"] is not None else "",
        snap["latency_ms_p95"] if snap["latency_ms_p95"] is not None else "",
        snap["latency_ms_p99"] if snap["latency_ms_p99"] is not None else "",
    ]

async def run_load(config: Config, rpm: int, duration_sec: int, concurrency: int, insecure_tls: bool, login_timeout: int, show_progress_every: int, stats_dir: Path):
    stats_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    progress_csv = stats_dir / f"load_progress_{ts}.csv"
    summary_csv  = stats_dir / f"load_summary_{ts}.csv"

    print(f"[INFO] Logging in {len(config.users)} users…")
    login_tasks = [login_user(config.base_url, config.login_path, cred, insecure_tls, login_timeout) for cred in config.users]
    sessions = [s for s in await asyncio.gather(*login_tasks) if s is not None]
    if not sessions:
        print("[FATAL] No successful logins; aborting.")
        return
    print(f"[INFO] {len(sessions)}/{len(config.users)} users logged in successfully.")

    stats = Stats(target_rpm=rpm)
    job_q: asyncio.Queue = asyncio.Queue(maxsize=rpm * 2 if rpm > 0 else 1000)
    workers = [asyncio.create_task(worker(f"W{i+1}", config.base_url, job_q, stats)) for i in range(concurrency)]

    prod = asyncio.create_task(producer(job_q, sessions, config.urls, config.parameters, rpm, duration_sec))

    import csv
    with progress_csv.open("w", encoding="utf-8", newline="") as pf:
        pw = csv.writer(pf)
        pw.writerow(_csv_header())

        async def progress():
            while not prod.done():
                await asyncio.sleep(show_progress_every)
                snap = stats.snapshot()
                now_iso = time.strftime("%Y-%m-%dT%H:%M:%S")
                headers = ["Metric","Value"]
                rows = [
                    ("Elapsed (s)", snap["elapsed_sec"]),
                    ("Target RPM",  snap["target_rpm"]),
                    ("Observed RPM",snap["observed_rpm"]),
                    ("Total",       snap["total"]),
                    ("Success",     snap["success"]),
                    ("Failures",    snap["failures"]),
                    ("p50 (ms)",    snap["latency_ms_p50"]),
                    ("p95 (ms)",    snap["latency_ms_p95"]),
                    ("p99 (ms)",    snap["latency_ms_p99"]),
                ]
                print("\n[PROGRESS]\n" + _format_table(headers, rows))
                pw.writerow(_csv_row(now_iso, snap))
                pf.flush()

        prog = asyncio.create_task(progress())

        await prod
        await job_q.join()
        for _ in workers:
            await job_q.put((None, UrlTemplate(path="/"), {}))
        await asyncio.gather(*workers, return_exceptions=True)
        prog.cancel()

    for s in sessions:
        await s.close()

    snap = stats.snapshot()
    headers = ["Metric","Value"]
    rows = [
        ("Elapsed (s)", snap["elapsed_sec"]),
        ("Target RPM",  snap["target_rpm"]),
        ("Observed RPM",snap["observed_rpm"]),
        ("Total",       snap["total"]),
        ("Success",     snap["success"]),
        ("Failures",    snap["failures"]),
        ("p50 (ms)",    snap["latency_ms_p50"]),
        ("p95 (ms)",    snap["latency_ms_p95"]),
        ("p99 (ms)",    snap["latency_ms_p99"]),
    ]
    print("\n[RESULTS]\n" + _format_table(headers, rows))
    print("\n[RESULTS JSON]\n" + json.dumps(snap, indent=2))

    import csv
    with summary_csv.open("w", encoding="utf-8", newline="") as sf:
        sw = csv.writer(sf)
        sw.writerow(_csv_header())
        sw.writerow(_csv_row(time.strftime("%Y-%m-%dT%H:%M:%S"), snap))
    print(f"[INFO] Progress CSV: {progress_csv}")
    print(f"[INFO] Summary  CSV: {summary_csv}")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Moodle LMS load generator (async) with table output & CSV capture")
    p.add_argument("--config", required=True, help="Path to config.json")
    p.add_argument("--rpm", type=int, required=True, help="Requests per minute target")
    p.add_argument("--duration", type=int, required=True, help="Duration in seconds")
    p.add_argument("--concurrency", type=int, default=20, help="Number of concurrent workers")
    p.add_argument("--insecure", action="store_true", help="Ignore TLS verification (local/self-signed)")
    p.add_argument("--login-timeout", type=int, default=20, help="Seconds for login HTTP timeout")
    p.add_argument("--progress", type=int, default=30, help="Progress print interval (seconds)")
    p.add_argument("--stats-dir", default="stats", help="Directory to write CSV stats (default: stats)")
    return p.parse_args()

def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Config.from_dict(data)

if __name__ == "__main__":
    try:
        import aiohttp
    except Exception:
        print("This script requires the 'aiohttp' package. Install with: pip install aiohttp")
        raise

    args = parse_args()
    cfg = load_config(args.config)
    asyncio.run(
        run_load(
            cfg,
            rpm=args.rpm,
            duration_sec=args.duration,
            concurrency=args.concurrency,
            insecure_tls=args.insecure,
            login_timeout=args.login_timeout,
            show_progress_every=args.progress,
            stats_dir=Path(args.stats_dir),
        )
    )
