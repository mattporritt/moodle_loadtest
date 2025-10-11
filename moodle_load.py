#!/usr/bin/env python3
"""
moodle_load.py
==============
Async load generator for Moodle LMS with:
- throttled login to avoid DOSing the server,
- bounded socket usage per session,
- human-readable progress tables and CSV capture,
- separate error log for failed requests,
- optional "browser emulation" to fetch assets and make lightweight AJAX probes.

Notes
-----
* Latency percentiles (p50/p95/p99) are computed **only from successful requests**.
  Failures are still counted in throughput and are also recorded in `errors_*.csv`.
* In browser emulation mode, we count **each sub-request** (assets/AJAX) as its own
  request for success/failure and timing, not just the top-level page load.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import csv
import json
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Iterable, Set
from urllib.parse import urljoin, urlparse, urlunparse
from html import unescape
import aiohttp
from aiohttp import ClientSession, TCPConnector

# ---------------------------------------------------------------------------
# Constants & Regex
# ---------------------------------------------------------------------------

LOGIN_PATH: str = "/login/index.php"
DASHBOARD_PATH: str = "/my/"

# Hidden CSRF field on the login form.
_LOGINTOKEN_RE = re.compile(r'name="logintoken"\s+value="([^"]+)"')
# Locale-specific message for bad credentials on stock Moodle (English).
_INVALID_LOGIN_RE = re.compile(r"\bInvalid login\b", re.IGNORECASE)

# Extract candidate asset URLs (href/src) from HTML.
_HREF_SRC_RE = re.compile(
    r'''(?ix)
    \b(?:href|src)\s*=\s*  # attribute
    (?:
        "([^"]+)"             # double-quoted
        |'([^']+)'            # single-quoted
        |([^\s>]+)           # unquoted (rare)
    )
    '''
)
# Commonly, sesskey shows up in config blobs or query strings.
_SESSKEY_RE = re.compile(r'\bsesskey\b[\s:=]+"?([A-Za-z0-9]+)"?', re.IGNORECASE)
_SESSKEY_QS_RE = re.compile(r'\bsesskey=([A-Za-z0-9]+)\b', re.IGNORECASE)

# Lightweight, harmless AJAX probe payload (keeps session warm).
_AJAX_TOUCH_PAYLOAD = [{"methodname": "core_session_touch", "args": {}}]

# ---------------------------------------------------------------------------
# URL / HTML utilities
# ---------------------------------------------------------------------------
def _is_login_url(url: str) -> bool:
    """
    Return True if `url` resolves to Moodle's login page path.

    Parameters
    ----------
    url : str
        Absolute or relative URL as observed after an HTTP request.

    Returns
    -------
    bool
        True if the URL path ends with LOGIN_PATH (ignoring trailing slash).
    """
    try:
        path = urlparse(url).path or ""
        return path.rstrip("/").endswith(LOGIN_PATH.rstrip("/"))
    except Exception:
        # Be conservative; if we fail to parse, we can't assert it's *not* login.
        return False

async def _fetch_logintoken(session: aiohttp.ClientSession, base_url: str) -> Optional[str]:
    """
    Retrieve the login page and extract Moodle's hidden 'logintoken'.

    Parameters
    ----------
    session : aiohttp.ClientSession
        Session used for HTTP requests.
    base_url : str
        Base site URL without trailing slash (e.g., 'https://moodle.example.com').

    Returns
    -------
    Optional[str]
        The logintoken if present, otherwise None (themes/flows may omit it).
    """
    login_url = urljoin(base_url, LOGIN_PATH)
    async with session.get(login_url, allow_redirects=True) as r:
        html = await r.text()
    m = _LOGINTOKEN_RE.search(html)
    return m.group(1) if m else None

async def _probe_logged_in(session: aiohttp.ClientSession, base_url: str) -> bool:
    """
    Probe a post-login page to confirm authentication.

    Strategy
    --------
    * GET /my/ (user dashboard)
    * If redirected to login (or URL is login), then auth failed.
    * Otherwise, consider it success. A weak heuristic (`logout.php`) is also
      used as a sanity hint but is not relied upon exclusively.

    Parameters
    ----------
    session : aiohttp.ClientSession
        Session used for HTTP requests.
    base_url : str
        Base site URL without trailing slash.

    Returns
    -------
    bool
        True if the user appears to be authenticated; False otherwise.
    """
    dash = urljoin(base_url, DASHBOARD_PATH)
    async with session.get(dash, allow_redirects=True) as r:
        final_url = str(r.url)
        body = await r.text()
    if _is_login_url(final_url):
        return False
    # Secondary hint (not authoritative across all themes):
    # when logged-in, Moodle pages often include a logout link.
    return ("logout.php" in body) or (not _is_login_url(final_url))

def _same_origin(url_a: str, url_b: str) -> bool:
    """Return True if `url_a` and `url_b` share scheme+netloc (origin)."""
    pa, pb = urlparse(url_a), urlparse(url_b)
    return (pa.scheme, pa.netloc) == (pb.scheme, pb.netloc)

def _normalize_asset_url(base_url: str, raw: str) -> Optional[str]:
    """
    Resolve `raw` relative to base and discard javascript:, mailto:, data:, tel:, and fragments.
    Also HTML-unescape and normalize backslash-escaped slashes so strings like
    '\\/\\/en.wikipedia.org\\/wiki\\/X' are handled correctly.

    This function also treats protocol-relative URLs (//host/path) properly and
    skips any URL that still contains quotes or obvious garbage after cleanup.
    """
    if not raw:
        return None

    # 1) Decode HTML entities and trim whitespace
    raw = unescape(raw).strip()

    # 2) Strip wrapping quotes if present
    if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
        raw = raw[1:-1].strip()

    # 3) Normalize common escape sequences from inline JSON/JS
    #    e.g. 'https:\\/\\/example.com\\/x' -> 'https://example.com/x'
    raw = raw.replace(r"\/", "/").replace(r"\'", "'").replace(r"\\#", "#").replace(r"\\:", ":")

    # 4) Drop schemes/targets we never want to fetch
    if raw.startswith(("#", "javascript:", "data:", "mailto:", "tel:")):
        return None

    # 5) Handle protocol-relative and "almost protocol-relative" (e.g. '/\/\/host/path')
    #    After step 3, things like '\\/\\/host' are now '//host'.
    scheme = urlparse(base_url).scheme or "https"
    stripped = raw.lstrip("/")  # remove leading slashes to detect '//host'
    if stripped.startswith("//"):
        # true (or almost-true) protocol-relative URL
        u = f"{scheme}://{stripped.lstrip('/')}"
    elif raw.startswith(("http://", "https://")):
        u = raw
    else:
        # relative URL – resolve against base
        u = urljoin(base_url + "/", raw)

    # 6) Defensive: drop URLs with quotes or remaining HTML entities
    if any(q in u for q in ['"', "'", "&quot;"]):
        return None

    return u

def extract_asset_urls(html: str, base_url: str, exclude_re: Optional[re.Pattern], same_origin_only: bool = True) -> List[str]:
    """Extract candidate asset URLs from HTML content."""
    urls: List[str] = []
    seen: Set[str] = set()

    for m in _HREF_SRC_RE.finditer(html):
        raw = m.group(1) or m.group(2) or m.group(3)
        u = _normalize_asset_url(base_url, raw)
        if not u:
            continue

        # Apply same-origin filter AFTER normalization so externals (wikipedia/creativecommons)
        # remain external and are skipped when same_origin_only=True.
        if same_origin_only and not _same_origin(base_url, u):
            continue

        if exclude_re and exclude_re.search(u):
            continue

        if u not in seen:
            seen.add(u)
            urls.append(u)
    return urls

def maybe_extract_sesskey(html: str) -> Optional[str]:
    """Attempt to extract Moodle sesskey from HTML blobs or query strings."""
    m = _SESSKEY_RE.search(html) or _SESSKEY_QS_RE.search(html)
    return m.group(1) if m else None

# ---------------------------------------------------------------------------
# Pretty table
# ---------------------------------------------------------------------------
def _format_table(headers: List[str], rows: List[Tuple[Any, Any]]) -> str:
    """
    Render a simple monospace table from headers + rows.

    We avoid third-party dependencies for portability; padding is calculated
    to align columns for human readability.

    Parameters
    ----------
    headers : List[str]
        Column headers to display.
    rows : List[Tuple[Any, Any]]
        A list of 2-tuples (Metric, Value).

    Returns
    -------
    str
        Multiline string with an ASCII table.
    """
    cols = len(headers)
    widths = [len(h) for h in headers]
    for row in rows:
        for i in range(cols):
            widths[i] = max(widths[i], len(str(row[i]) if i < len(row) else ""))

    # Separator mirrors column widths (+2 for padding around each cell)
    sep = "+".join("-" * (w + 2) for w in widths)
    lines = []
    header_line = " | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers))
    lines.append(header_line)
    lines.append(sep)
    for row in rows:
        line = " | ".join(str(row[i]).ljust(widths[i]) for i in range(cols))
        lines.append(line)
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Data models / config
# ---------------------------------------------------------------------------
@dataclass
class UserCred:
    """
    Credentials for a single test user.
    """
    username: str
    password: str

@dataclass
class UrlTemplate:
    """
    A single URL template to be requested during the test.

    Attributes
    ----------
    path : str
        Relative path (e.g., '/course/view.php?id={courseid}'). Placeholders
        in braces are resolved using `parameters` (see `ParamRange`).
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
    base_url : str
        Root URL for the Moodle site, without trailing slash.
    login_path : str
        Path to Moodle login (defaults to '/login/index.php').
    users : List[UserCred]
        Test user pool to cycle through for requests.
    urls : List[UrlTemplate]
        URL templates to hit; may include integer placeholders.
    parameters : Dict[str, ParamRange]
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

        Parameters
        ----------
        d : Dict[str, Any]
            Parsed JSON object (output of json.load).

        Returns
        -------
        Config
            Fully constructed config with trailing slashes normalized.
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

# ---------------------------------------------------------------------------
# Stats aggregation
# ---------------------------------------------------------------------------
@dataclass
class Stats:
    """
    Single-thread (event-loop) aggregator for request outcomes and latencies.

    Notes
    -----
    * `latencies` stores per-request durations **in seconds**.
    * Percentiles are computed naïvely at snapshot time (adequate for typical
      load-test scales; avoids external deps).
    """
    started_at: float = field(default_factory=time.monotonic)
    total_requests: int = 0
    success: int = 0
    failures: int = 0
    latencies: List[float] = field(default_factory=list)
    target_rpm: int = 0  # interpreted as "initiated requests per minute"

    def record(self, ok: bool, latency: float) -> None:
        """
        Record a single request outcome.

        Parameters
        ----------
        ok : bool
            True if HTTP status was 2xx/3xx, False otherwise.
        latency : float
            Request duration in seconds (includes redirects and body drain).
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

        Returns
        -------
        Dict[str, Any]
            Summary including elapsed seconds, RPM, totals, and p50/p95/p99
            in milliseconds (None when no successful samples yet).
        """
        elapsed = max(0.0001, time.monotonic() - self.started_at)
        rpm = self.total_requests / (elapsed / 60.0)
        p50, p95, p99 = self._percentiles([50, 95, 99])
        return {
            "elapsed_sec": round(elapsed, 1),
            "target_initiated_rpm": self.target_rpm,  # renamed for clarity
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
        Compute percentiles from the successful-request latencies.

        Parameters
        ----------
        ps : List[int]
            Percentile integers (e.g., [50, 95, 99]).

        Returns
        -------
        Tuple[Optional[int], Optional[int], Optional[int]]
            Percentiles in **milliseconds** (or Nones if no samples yet).
        """
        if not self.latencies:
            return (None, None, None)
        xs = sorted(self.latencies)
        out: List[int] = []
        for p in ps:
            # Index rounding strategy keeps endpoints stable for small samples.
            k = max(0, min(len(xs) - 1, int(round((p / 100.0) * (len(xs) - 1)))))
            out.append(int(xs[k] * 1000))
        return tuple(out)  # type: ignore[return-value]

# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------
def render_path(tpl: str, parameters: Dict[str, ParamRange]) -> str:
    """
    Resolve integer placeholders inside a URL template using configured ranges.

    Example
    -------
    '/user/profile.php?id={userid}' => '/user/profile.php?id=4321'

    Parameters
    ----------
    tpl : str
        Template path containing {placeholders}.
    parameters : Dict[str, ParamRange]
        Mapping of placeholder name to integer range.

    Returns
    -------
    str
        Path with placeholders replaced by random integers within the ranges.
    """
    def repl(m: re.Match[str]) -> str:
        name = m.group(1)
        if name not in parameters:
            raise ValueError(f"No parameter range defined for {{{name}}} in URL '{tpl}'")
        pr = parameters[name]
        return str(random.randint(pr.min, pr.max))
    return re.sub(r"\{([a-zA-Z0-9_]+)\}", repl, tpl)

async def login_user(
    session: aiohttp.ClientSession,
    base_url: str,
    username: str,
    password: str,
) -> bool:
    # Step 1: fetch token
    logintoken = await _fetch_logintoken(session, base_url)
    # Step 2: post creds
    form: Dict[str, Any] = {"username": username, "password": password, "rememberusername": 1}
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
    # Store sesskey if we can scrape one during login; helpful for later.
    sk = maybe_extract_sesskey(body)
    if sk:
        setattr(session, "sesskey", sk)
    return await _probe_logged_in(session, base_url)

async def login_and_return_session(
    base_url: str,
    login_path: str,  # present for compatibility with existing calls
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
            setattr(session, "username", cred.username)  # used for error logging
            return session
        await session.close()
        return None
    except Exception as e:
        with contextlib.suppress(Exception):
            await session.close()
        print(f"[LOGIN] {cred.username}: exception during login: {e}")
        return None

# ---------------------------------------------------------------------------
# Worker & producer
# ---------------------------------------------------------------------------
async def _fetch_one(
    session: ClientSession,
    url: str,
    stats: Stats,
    name: str,
    errors_writer: Optional[csv.writer],
    errors_lock: Optional[asyncio.Lock],
) -> None:
    """Fetch a single URL, record timing, and log a failure if needed."""
    t0 = time.monotonic()
    ok = False
    status: Optional[int] = None
    error_msg = ""
    try:
        async with session.get(url, allow_redirects=True) as resp:
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
                getattr(session, "username", "unknown"),
                url,
                status if status is not None else "",
                error_msg,
            ])

async def _ajax_touch(session: ClientSession, base_url: str, stats: Stats, name: str,
                      errors_writer: Optional[csv.writer], errors_lock: Optional[asyncio.Lock]) -> None:
    """Make a minimal, benign AJAX call if a sesskey is available."""
    sesskey: Optional[str] = getattr(session, "sesskey", None)
    if not sesskey:
        return
    url = urljoin(base_url, f"/lib/ajax/service.php?sesskey={sesskey}")
    t0 = time.monotonic()
    ok = False
    status = None
    error_msg = ""
    try:
        async with session.post(url, json=_AJAX_TOUCH_PAYLOAD, headers={
            "Content-Type": "application/json",
            "X-Requested-With": "XMLHttpRequest",
        }) as resp:
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
                getattr(session, "username", "unknown"),
                url,
                status if status is not None else "",
                error_msg,
            ])

async def worker(
    name: str,
    base_url: str,
    job_q: "asyncio.Queue",
    stats: Stats,
    errors_writer: Optional[csv.writer] = None,
    errors_lock: Optional[asyncio.Lock] = None,
    emulate_browser: bool = False,
    exclude_re: Optional[re.Pattern] = None,
    asset_concurrency: int = 6,
) -> None:
    """Consume jobs; optionally emulate a browser by fetching assets and AJAX."""
    sem_assets = asyncio.Semaphore(asset_concurrency)

    async def fetch_asset(url: str) -> None:
        async with sem_assets:
            await _fetch_one(session, url, stats, name, errors_writer, errors_lock)

    while True:
        try:
            session, url_tpl, param_ranges = await job_q.get()
            if session is None:
                job_q.task_done()
                return

            path = render_path(url_tpl.path, param_ranges)
            full_url = f"{base_url}{path}"

            # 1) Fetch the primary page
            t0 = time.monotonic()
            ok = False
            status: Optional[int] = None
            body: Optional[str] = None
            try:
                async with session.get(full_url, allow_redirects=True) as resp:
                    status = resp.status
                    ok = 200 <= resp.status < 400
                    ctype = resp.headers.get("Content-Type", "")
                    if "html" in ctype.lower():
                        body = await resp.text()
                    else:
                        await resp.read()
            except Exception as e:
                ok = False
                body = None
                err = str(e)
            else:
                err = ""

            dt = time.monotonic() - t0
            stats.record(ok, dt)

            if not ok and errors_writer and errors_lock:
                async with errors_lock:
                    errors_writer.writerow([
                        time.strftime("%Y-%m-%dT%H:%M:%S"),
                        name,
                        getattr(session, "username", "unknown"),
                        full_url,
                        status if status is not None else "",
                        err,
                    ])

            # 2) If emulating a browser and the body is HTML, fetch same-origin assets (respect --exclude-url-pattern)
            # and perform a minimal AJAX 'touch' to better mimic browser behavior.
            if emulate_browser and body:
                # Extract and remember sesskey (used for AJAX)
                sk = maybe_extract_sesskey(body)
                if sk:
                    setattr(session, "sesskey", sk)

                # Extract asset URLs from the HTML and fetch them (same-origin only)
                base_for_assets = base_url
                assets = extract_asset_urls(body, base_for_assets, exclude_re, same_origin_only=True)

                # Fetch assets with limited concurrency
                await asyncio.gather(*(fetch_asset(u) for u in assets))

                # Optionally do a harmless AJAX touch call (counts as a request)
                await _ajax_touch(session, base_url, stats, name, errors_writer, errors_lock)

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
    duration_sec: int,
) -> None:
    """Pace requests into the queue to approximate the target RPM (initiated requests)."""
    interval = 60.0 / max(1, rpm)
    end = time.monotonic() + duration_sec
    while time.monotonic() < end:
        sess = random.choice(sessions)
        tpl = random.choice(url_templates)
        await job_q.put((sess, tpl, param_ranges))
        await asyncio.sleep(interval)

# ---------------------------------------------------------------------------
# CSV utils
# ---------------------------------------------------------------------------
def _csv_header() -> List[str]:
    """
    Return the CSV header shared by progress and summary files. Uses 'target_initiated_rpm' terminology.
    """
    return [
        "timestamp",
        "elapsed_sec",
        "target_initiated_rpm",
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
    Serialize a Stats snapshot with a timestamp prefix for CSV writing.
    """
    return [
        now_iso,
        snap["elapsed_sec"],
        snap["target_initiated_rpm"],
        snap["observed_rpm"],
        snap["total"],
        snap["success"],
        snap["failures"],
        snap["latency_ms_p50"] if snap["latency_ms_p50"] is not None else "",
        snap["latency_ms_p95"] if snap["latency_ms_p95"] is not None else "",
        snap["latency_ms_p99"] if snap["latency_ms_p99"] is not None else "",
    ]

# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
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
    connector_limit_per_host: int,
    emulate_browser: bool,
    exclude_url_pattern: Optional[str],
) -> None:
    stats_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    progress_csv = stats_dir / f"load_progress_{ts}.csv"
    summary_csv  = stats_dir / f"load_summary_{ts}.csv"
    errors_csv   = stats_dir / f"errors_{ts}.csv"

    # Pre-compile exclusion regex if provided
    exclude_re = re.compile(exclude_url_pattern) if exclude_url_pattern else None

    # ----- 1) Login phase
    print(f"[INFO] Logging in {len(config.users)} users…")
    sem = asyncio.Semaphore(max(1, login_concurrency))

    async def login_one(cred: UserCred) -> Optional[ClientSession]:
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

    # ----- 2) Writers BEFORE starting workers
    with progress_csv.open("w", encoding="utf-8", newline="") as pf,          errors_csv.open("w", encoding="utf-8", newline="") as ef:

        pw = csv.writer(pf)
        pw.writerow(_csv_header())

        ew = csv.writer(ef)
        ew.writerow(["timestamp", "worker", "username", "url", "status", "error"])
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
                emulate_browser=emulate_browser,
                exclude_re=exclude_re,
            ))
            for i in range(concurrency)
        ]

        prod = asyncio.create_task(
            producer(job_q, sessions, config.urls, config.parameters, rpm, duration_sec)
        )

        async def progress_task() -> None:
            while not prod.done():
                await asyncio.sleep(show_progress_every)
                snap = stats.snapshot()
                now_iso = time.strftime("%Y-%m-%dT%H:%M:%S")
                headers = ["Metric", "Value"]
                rows = [
                    ("Elapsed (s)", snap["elapsed_sec"]),
                    ("Target RPM (initiated)",  snap["target_initiated_rpm"]),
                    ("Observed RPM", snap["observed_rpm"]),
                    ("Total", snap["total"]),
                    ("Success", snap["success"]),
                    ("Failures", snap["failures"]),
                    ("p50 (ms)", snap["latency_ms_p50"]),
                    ("p95 (ms)", snap["latency_ms_p95"]),
                    ("p99 (ms)", snap["latency_ms_p99"]),
                ]
                print("\n[PROGRESS]\n" + _format_table(headers, rows))
                pw.writerow(_csv_row(now_iso, snap))
                pf.flush()

        prog = asyncio.create_task(progress_task())

        # Wait for producer & workers to finish
        await prod
        await job_q.join()

        # Tell workers to shut down
        for _ in workers:
            await job_q.put((None, UrlTemplate(path="/"), {}))
        await asyncio.gather(*workers, return_exceptions=True)

        # Cancel progress task cleanly
        prog.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await prog

        # Final snapshot + summary CSV
        final_snap = stats.snapshot()
        headers = ["Metric", "Value"]
        rows = [
            ("Elapsed (s)", final_snap["elapsed_sec"]),
            ("Target RPM (initiated)",  final_snap["target_initiated_rpm"]),
            ("Observed RPM",final_snap["observed_rpm"]),
            ("Total",       final_snap["total"]),
            ("Success",     final_snap["success"]),
            ("Failures",    final_snap["failures"]),
            ("p50 (ms)",    final_snap["latency_ms_p50"]),
            ("p95 (ms)",    final_snap["latency_ms_p95"]),
            ("p99 (ms)",    final_snap["latency_ms_p99"]),
        ]
        print("\n[RESULTS]\n" + _format_table(headers, rows))

        with summary_csv.open("w", encoding="utf-8", newline="") as sf:
            sw = csv.writer(sf)
            sw.writerow(_csv_header())
            sw.writerow(_csv_row(time.strftime("%Y-%m-%dT%H:%M:%S"), final_snap))

    # ----- 4) Close sessions
    for s in sessions:
        with contextlib.suppress(Exception):
            await s.close()

    print(f"[INFO] Progress CSV: {progress_csv}")
    print(f"[INFO] Errors  CSV:  {errors_csv}")
    print(f"[INFO] Summary CSV:  {summary_csv}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments. '--rpm' represents initiated (top-level) requests per minute.
    Use '--emulate-browser' to fetch assets and do minimal AJAX; use '--exclude-url-pattern' to skip URLs.
    """
    p = argparse.ArgumentParser(
        description="Moodle LMS load generator (async) with throttled login, CSV output, and optional browser emulation"
    )
    p.add_argument("--config", required=True, help="Path to config.json")
    p.add_argument("--rpm", type=int, required=True, help="Initiated requests per minute (top-level)")
    p.add_argument("--duration", type=int, required=True, help="Duration in seconds")
    p.add_argument("--concurrency", type=int, default=20, help="Number of concurrent workers")
    p.add_argument("--insecure", action="store_true", help="Ignore TLS verification (local/self-signed)")
    p.add_argument("--login-timeout", type=int, default=20, help="Seconds for login HTTP timeout")
    p.add_argument("--progress", type=int, default=30, help="Progress print interval (seconds)")
    p.add_argument("--stats-dir", default="stats", help="Directory to write CSV stats (default: stats)")
    p.add_argument("--login-concurrency", type=int, default=20, help="Max concurrent login attempts (default: 20)")
    p.add_argument("--connector-limit", type=int, default=8, help="Max simultaneous connections per session (default: 8)")
    p.add_argument("--connector-limit-per-host", type=int, default=4, help="Max per-host connections per session (default: 4)")
    p.add_argument("--emulate-browser", action="store_true", default=False,
                   help="If set, fetch asset URLs referenced by HTML and make a minimal AJAX probe.")
    p.add_argument("--exclude-url-pattern", default=None,
                   help="Regex; URLs matching this pattern are not executed/downloaded in browser emulation mode.")
    return p.parse_args()

def load_config(path: str) -> "Config":
    """
    Load config.json from disk and construct a Config instance.
    """
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
            emulate_browser=args.emulate_browser,
            exclude_url_pattern=args.exclude_url_pattern,
        )
    )
