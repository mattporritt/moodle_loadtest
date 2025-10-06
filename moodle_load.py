#!/usr/bin/env python3
"""
moodle_load.py
==============
Async load generator for Moodle LMS.

Purpose
-------
This script logs in a pool of Moodle users and repeatedly requests a set of URL
templates at a target requests-per-minute (RPM) rate for a fixed duration.
It prints human-readable progress/summary tables to the terminal *and* writes
CSV snapshots/summary for later analysis (useful for git-bisect runs).

Key characteristics
-------------------
- Uses aiohttp for concurrency and connection pooling
- Handles Moodle login (logintoken + MoodleSession cookie check)
- Supports URL templates with integer placeholders (e.g. '/course/view.php?id={courseid}')
- Enforces approximate RPM by pacing a producer; workers pull from an async queue
- Emits periodic progress and an end-of-run summary as both tables and CSV rows

Typical usage
-------------
python moodle_load.py --config config.json --rpm 600 --duration 600 \
  --concurrency 30 --stats-dir stats --insecure
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from aiohttp import ClientSession, TCPConnector

# Regex to extract Moodle's hidden login token from the login form
LOGIN_TOKEN_RE = re.compile(r'name="logintoken"\s+value="([^"]+)"')


# ---------- Pretty table helpers ----------
def _format_table(headers: List[str], rows: List[Tuple[Any, Any]]) -> str:
    """
    Render a simple monospace table from headers + rows.

    We avoid third-party dependencies for portability; padding is calculated
    to align columns for human readability.
    """
    cols = len(headers)
    widths = [len(h) for h in headers]
    for row in rows:
        for i in range(cols):
            widths[i] = max(widths[i], len(str(row[i]) if i < len(row) else ""))

    # separator length mirrors column widths (+2 for padding around each cell)
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
    """Simple structure to hold a test user's username and password."""

    username: str
    password: str


@dataclass
class UrlTemplate:
    """
    A single URL template to be requested during the test.
    Example: "/course/view.php?id={courseid}"
    """

    path: str


@dataclass
class ParamRange:
    """
    Closed integer range for parameter substitution: [min, max].
    """

    min: int
    max: int


@dataclass
class Config:
    """
    In-memory representation of the JSON configuration.

    Attributes
    ----------
    base_url: str
        Root URL for the Moodle site, without trailing slash.
    login_path: str
        Path to Moodle login (defaults to '/login/index.php').
    users: List[UserCred]
        Test user pool to cycle through for requests.
    urls: List[UrlTemplate]
        URL templates to hit; may include integer placeholders.
    parameters: Dict[str, ParamRange]
        Integer ranges used to resolve placeholders in URLs.
    """

    base_url: str
    login_path: str
    users: List[UserCred]
    urls: List[UrlTemplate]
    parameters: Dict[str, ParamRange]

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Config":
        """
        Build a Config from a parsed JSON dict with basic normalization.
        """
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
    """
    Thread-safe-ish (single-loop) aggregator for request outcomes and latencies.

    Notes
    -----
    - `latencies` stores per-request durations in seconds.
    - Percentiles are computed naïvely at snapshot time; for load-test scale
      here that's fine and keeps the implementation simple.
    """
    started_at: float = field(default_factory=time.monotonic)
    total_requests: int = 0
    success: int = 0
    failures: int = 0
    latencies: List[float] = field(default_factory=list)
    target_rpm: int = 0

    def record(self, ok: bool, latency: float) -> None:
        """
        Record a single request outcome.

        Parameters
        ----------
        ok : bool
            True if HTTP status was 2xx/3xx, False otherwise.
        latency : float
            Request duration in seconds.
        """
        self.total_requests += 1
        if ok:
            self.success += 1
            self.latencies.append(latency)
        else:
            self.failures += 1

    def snapshot(self) -> Dict[str, Any]:
        """
        Produce an immutable view of current counters and percentiles.
        Returned data is suitable for JSON serialization and CSV rows.
        """
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
        """
        Return (p50, p95, p99) latencies in *milliseconds*. If no samples yet,
        returns (None, None, None).
        """
        if not self.latencies:
            return (None, None, None)
        xs = sorted(self.latencies)
        out: List[int] = []
        for p in ps:
            # index rounding strategy keeps endpoints stable for small samples
            k = max(0, min(len(xs) - 1, int(round((p / 100.0) * (len(xs) - 1)))))
            out.append(int(xs[k] * 1000))
        return tuple(out)  # type: ignore[return-value]


# ---------- Core helpers ----------
def render_path(tpl: str, parameters: Dict[str, ParamRange]) -> str:
    """
    Resolve integer placeholders inside a URL template using configured ranges.

    Example
    -------
    '/user/profile.php?id={userid}' => '/user/profile.php?id=4321'
    """
    def repl(m: re.Match[str]) -> str:
        name = m.group(1)
        if name not in parameters:
            raise ValueError(f"No parameter range defined for {{{name}}} in URL '{tpl}'")
        pr = parameters[name]
        return str(random.randint(pr.min, pr.max))

    return re.sub(r"\{([a-zA-Z0-9_]+)\}", repl, tpl)


async def fetch_logintoken(session: ClientSession, login_url: str) -> Optional[str]:
    """
    Fetch the Moodle login page and extract the hidden 'logintoken' field.

    Returns None if the token isn't found (custom themes or flows may omit it).
    """
    async with session.get(login_url, allow_redirects=True) as resp:
        html = await resp.text()
        m = LOGIN_TOKEN_RE.search(html)
        return m.group(1) if m else None


async def login_user(
    base_url: str,
    login_path: str,
    cred: UserCred,
    insecure_tls: bool,
    timeout: int = 20,
) -> Optional[ClientSession]:
    """
    Create a dedicated session per user and attempt login.

    Logic
    -----
    - Request login page → parse logintoken
    - POST credentials (+ token if present)
    - Consider login successful if:
        * HTTP status is 200/302/303, and
        * a 'MoodleSession*' cookie is set
    - If login fails, close the session to free sockets.
    """
    connector = TCPConnector(ssl=False) if insecure_tls else TCPConnector(ssl=None)
    session = ClientSession(
        connector=connector,
        timeout=aiohttp.ClientTimeout(total=timeout),
        headers={"User-Agent": "moodle-loadgen/1.2"},
        cookie_jar=aiohttp.CookieJar(unsafe=insecure_tls),
    )
    try:
        login_url = f"{base_url}{login_path}"
        token = await fetch_logintoken(session, login_url)
        if not token:
            # Non-fatal: some themes/flows don't include tokens.
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


async def worker(
    name: str,
    base_url: str,
    job_q: "asyncio.Queue[Tuple[ClientSession, UrlTemplate, Dict[str, ParamRange]]]",
    stats: Stats,
) -> None:
    """
    Consume jobs from the queue and perform HTTP GETs.

    Worker lifecycle
    ----------------
    - Blocks on `job_q.get()` until a job arrives.
    - Receives a (session, url_template, param_ranges) tuple.
    - Renders the URL, fires the request, records latency & success.
    - Stops when it receives a 'poison pill' job with session=None.
    """
    while True:
        try:
            session, url_tpl, param_ranges = await job_q.get()
            if session is None:
                # 'Poison pill' to shut down the worker gracefully.
                job_q.task_done()
                return

            # Resolve placeholders for this request instance.
            path = render_path(url_tpl.path, param_ranges)
            full_url = f"{base_url}{path}"

            t0 = time.monotonic()
            ok = False
            try:
                # Allow redirects; we count 2xx/3xx as success to keep signal clean
                async with session.get(full_url, allow_redirects=True) as resp:
                    ok = 200 <= resp.status < 400
                    # Drain body to ensure the connection can be reused by aiohttp
                    await resp.read()
            except Exception as e:
                ok = False
                print(f"[{name}] Request error: {e} :: {full_url}")

            dt = time.monotonic() - t0
            stats.record(ok, dt)
            job_q.task_done()

        except asyncio.CancelledError:
            # Task cancelled by coordinator: exit silently
            return
        except Exception as e:
            print(f"[{name}] Unexpected worker error: {e}")


async def producer(
    job_q: "asyncio.Queue[Tuple[ClientSession, UrlTemplate, Dict[str, ParamRange]]]",
    sessions: List[ClientSession],
    url_templates: List[UrlTemplate],
    param_ranges: Dict[str, ParamRange],
    rpm: int,
    duration_sec: int,
) -> None:
    """
    Pace requests into the queue to approximate the target RPM.

    Strategy
    --------
    - Compute inter-arrival time: 60 / RPM seconds per request.
    - On each tick, enqueue a job selecting a random user session and template.
    - Stop after the specified duration.
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
    """
    CSV header used by both progress snapshots and final summary.
    """
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


def _csv_row(now_iso: str, snap: Dict[str, Any]) -> List[Any]:
    """
    Serialize a stats snapshot into a CSV row with a timestamp prefix.
    """
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


# ---------- Orchestration ----------
async def run_load(
    config: Config,
    rpm: int,
    duration_sec: int,
    concurrency: int,
    insecure_tls: bool,
    login_timeout: int,
    show_progress_every: int,
    stats_dir: Path,
) -> None:
    """
    End-to-end coordinator.

    Steps
    -----
    1) Login all users to separate sessions (for isolated cookies).
    2) Launch N workers that consume jobs and make requests.
    3) Start a producer that enqueues jobs at the chosen RPM.
    4) Periodically print a progress table and append a row to progress CSV.
    5) After the producer stops and queue drains, print/write final summary.
    """
    # Prepare output files
    stats_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    progress_csv = stats_dir / f"load_progress_{ts}.csv"
    summary_csv = stats_dir / f"load_summary_{ts}.csv"

    # ----- 1) Login phase
    print(f"[INFO] Logging in {len(config.users)} users…")
    login_tasks = [
        login_user(config.base_url, config.login_path, cred, insecure_tls, login_timeout)
        for cred in config.users
    ]
    sessions = [s for s in await asyncio.gather(*login_tasks) if s is not None]
    if not sessions:
        print("[FATAL] No successful logins; aborting.")
        return
    print(f"[INFO] {len(sessions)}/{len(config.users)} users logged in successfully.")

    # ----- 2) Worker pool
    stats = Stats(target_rpm=rpm)
    job_q: asyncio.Queue = asyncio.Queue(maxsize=rpm * 2 if rpm > 0 else 1000)
    workers = [
        asyncio.create_task(worker(f"W{i+1}", config.base_url, job_q, stats))
        for i in range(concurrency)
    ]

    # ----- 3) Producer
    prod = asyncio.create_task(
        producer(job_q, sessions, config.urls, config.parameters, rpm, duration_sec)
    )

    # Open CSV for periodic progress capture
    with progress_csv.open("w", encoding="utf-8", newline="") as pf:
        pw = csv.writer(pf)
        pw.writerow(_csv_header())

        async def progress_loop() -> None:
            """
            Periodically print a table and append a progress row to CSV.

            Runs until the producer signals completion.
            """
            while not prod.done():
                await asyncio.sleep(show_progress_every)
                snap = stats.snapshot()
                now_iso = time.strftime("%Y-%m-%dT%H:%M:%S")

                # Print readable table to terminal
                headers = ["Metric", "Value"]
                rows = [
                    ("Elapsed (s)", snap["elapsed_sec"]),
                    ("Target RPM", snap["target_rpm"]),
                    ("Observed RPM", snap["observed_rpm"]),
                    ("Total", snap["total"]),
                    ("Success", snap["success"]),
                    ("Failures", snap["failures"]),
                    ("p50 (ms)", snap["latency_ms_p50"]),
                    ("p95 (ms)", snap["latency_ms_p95"]),
                    ("p99 (ms)", snap["latency_ms_p99"]),
                ]
                print("\n[PROGRESS]\n" + _format_table(headers, rows))

                # Append to CSV and flush so tail -f shows updates
                pw.writerow(_csv_row(now_iso, snap))
                pf.flush()

        prog_task = asyncio.create_task(progress_loop())

        # ----- 4) Wait for producer completion & queue drain
        await prod
        await job_q.join()

        # Send 'poison pills' to workers to shut them down
        for _ in workers:
            await job_q.put((None, UrlTemplate(path="/"), {}))
        await asyncio.gather(*workers, return_exceptions=True)

        # Stop the progress reporter
        prog_task.cancel()

    # Close sessions after the loop ends
    for s in sessions:
        await s.close()

    # ----- 5) Final summary
    snap = stats.snapshot()
    headers = ["Metric", "Value"]
    rows = [
        ("Elapsed (s)", snap["elapsed_sec"]),
        ("Target RPM", snap["target_rpm"]),
        ("Observed RPM", snap["observed_rpm"]),
        ("Total", snap["total"]),
        ("Success", snap["success"]),
        ("Failures", snap["failures"]),
        ("p50 (ms)", snap["latency_ms_p50"]),
        ("p95 (ms)", snap["latency_ms_p95"]),
        ("p99 (ms)", snap["latency_ms_p99"]),
    ]
    print("\n[RESULTS]\n" + _format_table(headers, rows))
    print("\n[RESULTS JSON]\n" + json.dumps(snap, indent=2))

    # Write final row to a separate summary CSV
    with summary_csv.open("w", encoding="utf-8", newline="") as sf:
        sw = csv.writer(sf)
        sw.writerow(_csv_header())
        sw.writerow(_csv_row(time.strftime("%Y-%m-%dT%H:%M:%S"), snap))

    print(f"[INFO] Progress CSV: {progress_csv}")
    print(f"[INFO] Summary  CSV: {summary_csv}")


# ---------- CLI ----------
def parse_args() -> argparse.Namespace:
    """Define and parse CLI arguments."""
    p = argparse.ArgumentParser(
        description="Moodle LMS load generator (async) with table output & CSV capture"
    )
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
    """Load config JSON from disk and return a Config object."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Config.from_dict(data)


if __name__ == "__main__":
    # Surface a clearer error if aiohttp is missing
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
        )
    )
