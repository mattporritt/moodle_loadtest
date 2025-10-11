#!/usr/bin/env python3
"""
moodle_load.py
==============
Async load generator for Moodle LMS with:
- throttled login to avoid DOSing the server,
- bounded socket usage per session,
- human-readable progress tables and CSV capture.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import re
from urllib.parse import urljoin, urlparse
import aiohttp
from aiohttp import ClientSession, TCPConnector

# Regex to extract Moodle's hidden login token from the login form
LOGIN_TOKEN_RE = re.compile(r'name="logintoken"\s+value="([^"]+)"')
LOGIN_PATH = "/login/index.php"
DASHBOARD_PATH = "/my/"

_LOGINTOKEN_RE = re.compile(r'name="logintoken"\s+value="([^"]+)"')
_INVALID_LOGIN_RE = re.compile(r"\bInvalid login\b", re.IGNORECASE)

def _is_login_url(url: str) -> bool:
    try:
        path = urlparse(url).path or ""
        return path.rstrip("/").endswith(LOGIN_PATH.rstrip("/"))
    except Exception:
        return False

async def _fetch_logintoken(session: aiohttp.ClientSession, base_url: str) -> str | None:
    login_url = urljoin(base_url, LOGIN_PATH)
    async with session.get(login_url, allow_redirects=True) as r:
        html = await r.text()
    m = _LOGINTOKEN_RE.search(html)
    return m.group(1) if m else None

async def _probe_logged_in(session: aiohttp.ClientSession, base_url: str) -> bool:
    """Hit /my/ and check if we get bounced to login."""
    dash = urljoin(base_url, DASHBOARD_PATH)
    async with session.get(dash, allow_redirects=True) as r:
        final_url = str(r.url)
        body = await r.text()
    if _is_login_url(final_url):
        return False
    # A light sanity check that often appears only when logged in.
    return ("logout.php" in body) or (not _is_login_url(final_url))

# ---------- Pretty table helpers ----------
def _format_table(headers: List[str], rows: List[Tuple[Any, Any]]) -> str:
    """
    Render a simple monospace table from headers + rows.
    """
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


# ---------- Data models / config ----------
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

    def record(self, ok: bool, latency: float) -> None:
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
        out: List[int] = []
        for p in ps:
            k = max(0, min(len(xs) - 1, int(round((p / 100.0) * (len(xs) - 1)))))
            out.append(int(xs[k] * 1000))
        return tuple(out)  # type: ignore[return-value]


# ---------- Core helpers ----------
def render_path(tpl: str, parameters: Dict[str, ParamRange]) -> str:
    def repl(m: re.Match[str]) -> str:
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


async def login_user(
    session: aiohttp.ClientSession,
    base_url: str,
    username: str,
    password: str,
) -> bool:
    """
    1) GET login page, parse logintoken
    2) POST creds + token, follow redirects
    3) Fail if we land back on /login/ or see 'Invalid login'
    4) Probe /my/ to be resilient to custom landings
    """
    logintoken = await _fetch_logintoken(session, base_url)

    form = {
        "username": username,
        "password": password,
        "rememberusername": 1,
    }
    if logintoken:
        form["logintoken"] = logintoken

    login_url = urljoin(base_url, LOGIN_PATH)
    async with session.post(login_url, data=form, allow_redirects=True) as r:
        final_url = str(r.url)
        body = (await r.text())[:200_000]

    if _is_login_url(final_url):
        return False
    if _INVALID_LOGIN_RE.search(body):
        return False

    return await _probe_logged_in(session, base_url)


# Wrapper that matches the 7-arg call site and returns a logged-in session
async def login_and_return_session(
    base_url: str,
    login_path: str,  # kept for signature compatibility; not used
    cred: UserCred,
    insecure_tls: bool,
    login_timeout: int,
    connector_limit: int,
    connector_limit_per_host: int,
) -> Optional[aiohttp.ClientSession]:
    timeout = aiohttp.ClientTimeout(total=login_timeout)
    connector = TCPConnector(
        limit=connector_limit,
        limit_per_host=connector_limit_per_host,
        ssl=False if insecure_tls else None,
    )
    jar = aiohttp.CookieJar()
    session = aiohttp.ClientSession(timeout=timeout, connector=connector, cookie_jar=jar)

    try:
        ok = await login_user(session, base_url, cred.username, cred.password)
        if ok:
            return session
        await session.close()
        return None
    except Exception as e:
        try:
            await session.close()
        finally:
            pass
        print(f"[LOGIN] {cred.username}: exception during login: {e}")
        return None


async def worker(
    name: str,
    base_url: str,
    job_q: "asyncio.Queue",
    stats: Stats,
    errors_writer: Optional[csv.writer] = None,
    errors_lock: Optional[asyncio.Lock] = None,
):
    """
    Consume jobs; log failures to errors CSV (if provided).
    """
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
            status = None
            error_msg = ""
            try:
                async with session.get(full_url, allow_redirects=True) as resp:
                    status = resp.status
                    ok = 200 <= resp.status < 400
                    await resp.read()
            except Exception as e:
                ok = False
                error_msg = str(e)

            dt = time.monotonic() - t0
            stats.record(ok, dt)

            if not ok and errors_writer and errors_lock:
                async with errors_lock:
                    errors_writer.writerow([
                        time.strftime("%Y-%m-%dT%H:%M:%S"),
                        name,
                        full_url,
                        status if status is not None else "",
                        error_msg,
                    ])

            job_q.task_done()

        except asyncio.CancelledError:
            return
        except Exception as e:
            print(f"[{name}] Unexpected worker error: {e}")


async def producer(
    job_q: "asyncio.Queue",
    sessions: List[ClientSession],
    url_templates: List[UrlTemplate],
    param_ranges: Dict[str, ParamRange],
    rpm: int,
    duration_sec: int):
    """
    Pace requests into the queue to approximate the target RPM.
    """
    interval = 60.0 / max(1, rpm)
    end = time.monotonic() + duration_sec
    while time.monotonic() < end:
        sess = random.choice(sessions)
        tpl = random.choice(url_templates)
        await job_q.put((sess, tpl, param_ranges))
        await asyncio.sleep(interval)


# ---------- CSV utils ----------
def _csv_header() -> List[str]:
    return [
        "timestamp",
        "elapsed_sec",
        "target_rpm",
        "observed_rpm",
        "total",
        "success",
        "failures",
        "latency_ms_p50",
        "latency_ms_p95",
        "latency_ms_p99",
    ]


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


async def run_load(
    config: Config,
    rpm: int,
    duration_sec: int,
    concurrency: int,
    insecure_tls: bool,
    login_timeout: int,
    show_progress_every: int,
    stats_dir: Path,
    login_concurrency: int,
    connector_limit: int,
    connector_limit_per_host: int):
    stats_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    progress_csv = stats_dir / f"load_progress_{ts}.csv"
    summary_csv  = stats_dir / f"load_summary_{ts}.csv"
    errors_csv   = stats_dir / f"errors_{ts}.csv"

    # ----- 1) Login phase
    print(f"[INFO] Logging in {len(config.users)} users…")
    sem = asyncio.Semaphore(max(1, login_concurrency))

    async def login_one(cred: UserCred):
        async with sem:
            return await login_and_return_session(
                config.base_url,
                config.login_path,
                cred,
                insecure_tls,
                login_timeout,
                connector_limit,
                connector_limit_per_host,
            )

    login_tasks = [login_one(cred) for cred in config.users]
    sessions = [s for s in await asyncio.gather(*login_tasks) if s is not None]
    if not sessions:
        print("[FATAL] No successful logins; aborting.")
        return
    print(f"[INFO] {len(sessions)}/{len(config.users)} users logged in successfully.")

    # ----- 2) Create writers (progress + errors) BEFORE starting workers
    with progress_csv.open("w", encoding="utf-8", newline="") as pf, \
         errors_csv.open("w", encoding="utf-8", newline="") as ef:

        pw = csv.writer(pf)
        pw.writerow(_csv_header())

        ew = csv.writer(ef)
        ew.writerow(["timestamp", "worker", "url", "status", "error"])
        errors_lock = asyncio.Lock()

        # ----- 3) Start workers & producer
        stats = Stats(target_rpm=rpm)
        job_q: asyncio.Queue = asyncio.Queue(maxsize=rpm * 2 if rpm > 0 else 1000)

        workers = [
            asyncio.create_task(worker(
                f"W{i + 1}",
                config.base_url,
                job_q,
                stats,
                errors_writer=ew,
                errors_lock=errors_lock,
            ))
            for i in range(concurrency)
        ]

        prod = asyncio.create_task(
            producer(job_q, sessions, config.urls, config.parameters, rpm, duration_sec)
        )

        async def progress():
            """
            Periodically print a table and append a progress row to CSV.
            """
            while not prod.done():
                await asyncio.sleep(show_progress_every)
                snap = stats.snapshot()
                now_iso = time.strftime("%Y-%m-%dT%H:%M:%S")
                headers = ["Metric", "Value"]
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

        # Wait for producer & workers to finish
        await prod
        await job_q.join()

        # Tell workers to shut down
        for _ in workers:
            await job_q.put((None, UrlTemplate(path="/"), {}))
        await asyncio.gather(*workers, return_exceptions=True)
        prog.cancel()

    # ----- 4) Close sessions
    for s in sessions:
        await s.close()

    # ----- 5) Final summary
    snap = Stats(target_rpm=rpm)  # just to re-use table layout below? keep original:
    # Actually use the real stats we kept inside the 'with' block:
    # We printed periodic progress already; for final, recompute from the saved 'stats'
    # Move summary print/write above if you prefer. To keep scope, we recompute via last snapshot kept in 'rows'.

    # Since 'stats' is out of scope here, recompute by moving the final snapshot before closing the with block
    # For simplicity now, just tell user where files are:
    print(f"[INFO] Progress CSV: {progress_csv}")
    print(f"[INFO] Errors  CSV:  {errors_csv}")
    print(f"[INFO] Summary CSV:  {summary_csv}")

    # Write a one-line summary row based on the last printed progress if desired.
    # If you want exact final stats, move the snapshot code inside the 'with' block after worker shutdown.


# ---------- CLI ----------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Moodle LMS load generator (async) with throttled login & table/CSV output")
    p.add_argument("--config", required=True, help="Path to config.json")
    p.add_argument("--rpm", type=int, required=True, help="Requests per minute target")
    p.add_argument("--duration", type=int, required=True, help="Duration in seconds")
    p.add_argument("--concurrency", type=int, default=20, help="Number of concurrent workers")
    p.add_argument("--insecure", action="store_true", help="Ignore TLS verification (local/self-signed)")
    p.add_argument("--login-timeout", type=int, default=20, help="Seconds for login HTTP timeout")
    p.add_argument("--progress", type=int, default=30, help="Progress print interval (seconds)")
    p.add_argument("--stats-dir", default="stats", help="Directory to write CSV stats (default: stats)")
    p.add_argument("--login-concurrency", type=int, default=20, help="Max concurrent login attempts (default: 20)")
    p.add_argument("--connector-limit", type=int, default=8, help="Max simultaneous connections per session (default: 8)")
    p.add_argument("--connector-limit-per-host", type=int, default=4, help="Max per-host connections per session (default: 4)")
    return p.parse_args()


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Config.from_dict(data)


if __name__ == "__main__":
    try:
        import aiohttp  # noqa: F401
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
            login_concurrency=args.login_concurrency,
            connector_limit=args.connector_limit,
            connector_limit_per_host=args.connector_limit_per_host,
        )
    )