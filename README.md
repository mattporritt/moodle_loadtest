# Moodle LMS Load Generator (async)

A small Python tool to generate consistent load against a Moodle test site for performance diagnostics and git-bisect hunts. It logs in a pool of users, hits configured URLs (with optional random integer placeholders), paces to a target RPM, and reports latency percentiles.

## Features
- Username/password login against `/login/index.php` (handles `logintoken`)
- URL templates with `{param}` placeholders resolved to random ints from configured ranges
- Target requests-per-minute with async concurrency
- Duration-based run with periodic progress snapshots and final stats
- No JS/AJAX execution (by design for first-pass backend perf checks)

## Quick start
```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run for 10 minutes (600s) at 600 RPM with 30 workers:
python moodle_load.py   --config config.json   --rpm 600   --duration 600   --concurrency 30   --insecure
```

> Use `--insecure` for local/self-signed TLS. Omit it for proper certificates.

## Configuration
Create `config.json`:

```json
{
  "base_url": "https://moodle.test",
  "login_path": "/login/index.php",

  "users": [
    { "username": "perfuser01", "password": "pass01" },
    { "username": "perfuser02", "password": "pass02" }
  ],

  "parameters": {
    "courseid": { "min": 2, "max": 500 },
    "cmid":     { "min": 10, "max": 2000 },
    "userid":   { "min": 5, "max": 10000 }
  },

  "urls": [
    { "path": "/my/" },
    { "path": "/course/view.php?id={courseid}" },
    { "path": "/mod/forum/view.php?id={cmid}" },
    { "path": "/user/profile.php?id={userid}" }
  ]
}
```

- `base_url` + `path` => final URL.
- Any `{name}` in a path or query is replaced with a random int from `parameters[name]`.

## CLI
```
--config PATH          Path to config.json (required)
--rpm INT              Requests per minute target (required)
--duration INT         Duration in seconds (required)
--concurrency INT      Parallel workers (default: 20)
--insecure             Ignore TLS verification (self-signed/local)
--login-timeout INT    Login HTTP timeout seconds (default: 20)
--progress INT         Progress print interval seconds (default: 30)
```

## Output
- Progress snapshots every `--progress` seconds with observed RPM and latency p50/p95/p99.
- Final JSON summary printed to stdout, e.g.:
```json
{
  "elapsed_sec": 600.1,
  "total": 6000,
  "success": 5987,
  "failures": 13,
  "req_per_min_observed": 599.1,
  "latency_ms_p50": 120,
  "latency_ms_p95": 380,
  "latency_ms_p99": 640
}
```

## Tips
- Scale `--concurrency` up if the observed RPM is below target due to latency.
- Tune URL mix in `config.json` to focus on hot paths (course view, quiz, gradebook, dashboard).
- Ensure test users actually have access to the targeted pages (enrolments/permissions).

## Git bisect flow (suggested)
1. Snapshot current commit performance with a fixed config and RPM.
2. `git bisect start` → mark good/bad commits.
3. Rebuild/redeploy test site at each step, run the same command, record p95/p99 and observed RPM.
4. Narrow to the offending commit(s), then dive deeper (DB traces, PHP profiling, slow logs).

## Troubleshooting
- **0 users logged in**: Check credentials and that `/login/index.php` is accessible; some themes/plugins change login flow.
- **Observed RPM low**: Increase `--concurrency`, use faster client host, or reduce network latency.
- **Many 403/404**: Users may lack access, or URLs require different params.

## Roadmap (easy adds)
- Per-URL weights and per-endpoint metrics
- CSV/JSONL request logs
- Exponential backoff & retries on transient errors
- Ramp-up/warm-up phase
- Fixed RNG seed for reproducible runs
