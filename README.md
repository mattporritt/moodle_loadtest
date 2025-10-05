# Moodle LMS Load Generator (async)

A Python toolkit for generating **repeatable load** on a Moodle LMS test site to diagnose performance regressions (e.g., during `git bisect`). It includes:
- an async load generator,
- a password reset/population helper for test users (Postgres),
- and a Docker container resource capture script for CPU/memory/IO metrics.

---

## Repository Contents

- **`moodle_load.py`** – Async load generator (no JS/AJAX execution) that logs in a pool of users, requests configured URLs, and reports latency percentiles and observed RPM.
- **`set_moodle_passwords.py`** – Connects to a Postgres Moodle DB, resets passwords for a selected set of users to a known value, and writes those users to `config.json` (for the load generator).
- **`capture_docker_stats.py`** – Samples Docker container **CPU%**, **memory usage**, **PIDs**, **network bytes**, and **block I/O** at a fixed interval and writes them to CSVs (one file per container).
- **`requirements.txt`** – Python dependencies.

---

## Quick Start

### 1) Python environment

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

To **deactivate** later:
```bash
deactivate
```

### 2) Prepare the Moodle test site (seed → reset passwords → backup)

1. **Seed test data** (choose a size that suits: S/M/L):
   ```bash
   ./bin/moodle-docker-compose exec webserver      php public/admin/tool/generator/cli/maketestsite.php --size=M
   ```

2. **Reset passwords and populate config** (Postgres):
   ```bash
   # Adjust DB creds/host and desired count/password
   python set_moodle_passwords.py      --db-name moodle      --db-user moodleuser      --db-pass S3cret      --db-host localhost      --count 200      --password Passw0rd!      --config config.json
   ```
   This will:
   - bcrypt-hash the given password,
   - set `auth='manual'` and `confirmed=1` for those users,
   - and write a `users` block into `config.json`.

3. **Backup the database (pre‑test snapshot)**:
   ```bash
   pg_dump -U moodleuser -h localhost -F c -f pretest_backup.dump moodle
   ```

### 3) Configure load

Create or edit **`config.json`**:

```json
{
  "base_url": "https://moodle.test",
  "login_path": "/login/index.php",
  "users": [
    { "username": "perfuser01", "password": "Passw0rd!" },
    { "username": "perfuser02", "password": "Passw0rd!" }
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

- `base_url` + `path` → final URL.
- `{param}` placeholders are replaced by random integers within the configured range each request.

### 4) Run a load test

```bash
python moodle_load.py   --config config.json   --rpm 600   --duration 600   --concurrency 30   --insecure
```

You’ll see progress snapshots every 30s and a final JSON summary.

### 5) Capture Docker CPU & Memory during the run (optional but recommended)

If you use Docker (e.g., Moodle Docker), capture resource usage for the relevant containers while the test runs.

**Install the Docker SDK for Python** (already in `requirements.txt`? if not, run):
```bash
pip install docker
```

**Run the capture** (example for two containers):
```bash
python capture_docker_stats.py   --containers moodlemaster-webserver-1 moodlemaster-db-1   --interval 1   --duration 600   --outdir stats   --tag $(git rev-parse --short HEAD)   --human
```

Outputs:
```
stats/moodlemaster-webserver-1_YYYYmmdd-HHMMSS_<tag>.csv
stats/moodlemaster-db-1_YYYYmmdd-HHMMSS_<tag>.csv
```
CSV columns: `timestamp, container, cpu_percent, mem_used_bytes, mem_limit_bytes, pids, net_rx_bytes, net_tx_bytes, blk_read_bytes, blk_write_bytes` (+ human-readable columns when `--human` is used).

> Alternative (bash one-liner) using `docker stats` directly (human-formatted values):
> ```bash
> outfile=container_stats.csv
> echo "ts,container,name,cpu_perc,mem_usage,mem_perc,net_io,block_io,pids" > "$outfile"
> while :; do
>   docker stats moodlemaster-webserver-1 moodlemaster-db-1 --no-stream >     --format '{{.Container}},{{.Name}},{{.CPUPerc}},{{.MemUsage}},{{.MemPerc}},{{.NetIO}},{{.BlockIO}},{{.PIDs}}' >   | awk -v d="$(date +%FT%T)" '{print d","$0}' >> "$outfile"
>   sleep 1
> done
> ```

### 6) Restore the site to pre‑test state

```bash
pg_restore -U moodleuser -h localhost -d moodle -c pretest_backup.dump
```

---

## Command Reference

### `moodle_load.py`

```
--config PATH          Path to config.json (required)
--rpm INT              Requests per minute target (required)
--duration INT         Duration in seconds (required)
--concurrency INT      Parallel workers (default: 20)
--insecure             Ignore TLS verification (self-signed/local)
--login-timeout INT    Login HTTP timeout in seconds (default: 20)
--progress INT         Progress print interval seconds (default: 30)
```

### `set_moodle_passwords.py`

```
--db-name NAME         Database name (required)
--db-user USER         Database user (required)
--db-pass PASS         Database password (required)
--db-host HOST         Database host (default: localhost)
--prefix STR           Table prefix (default: mdl_)
--count INT            Users to update (default: 100)
--password STR         New password (required)
--config PATH          Path to load generator config.json (default: config.json)
--where SQL            Optional SQL filter (no leading WHERE)
--dry-run              Preview updates without modifying DB
--bcrypt-cost INT      bcrypt cost factor (default: 12)
```

### `capture_docker_stats.py`

```
--containers NAMES...  Container names/IDs (space-separated)
--interval FLOAT       Sampling interval seconds (default: 1.0)
--duration INT         Total duration seconds (default: 600)
--outdir PATH          Output directory for CSVs (default: stats)
--tag STR              Optional tag to include in filenames (e.g., git hash)
--human                Add human-readable columns to CSV
```

---

## Understanding Output & Errors

During a run, you’ll see a mix of INFO/WARN/ERROR and progress lines. Here’s what the common messages mean:

| Message snippet | What it means | What to check |
| --- | --- | --- |
| `[INFO] Logging in N users…` | Starting authentication for test users | OK |
| `[WARN] No logintoken for <user>` | Login form didn’t include the hidden token | Custom theme? Different login URL? Check `login_path` |
| `[ERROR] Login failed for <user>` | Credentials invalid or `auth` method mismatch | Ensure password reset ran; `auth='manual'` |
| `[W#] Request error: <exception>` | Network error, timeout, DNS, connection reset | Server reachable? Increase `--concurrency` only after connectivity is stable |
| `2xx/3xx counted as success` | Non-error HTTP responses are considered success | OK |
| Many `403/404` responses | Access denied / missing enrolment / wrong URLs | Ensure users have the right enrolments/permissions |
| `[PROGRESS] {...}` | Periodic snapshot | Look at `req_per_min_observed` and latency percentiles |
| `req_per_min_observed` lower than target | Client couldn’t sustain the target RPM | Raise `--concurrency`, reduce latency to server, or run client on faster host |
| High `latency_ms_p95/p99` | Tail latency is high | Potential server bottleneck (DB, PHP-FPM, caching, I/O) |
| Frequent failures/timeouts | Overloaded server or network instability | Lower RPM, add caching, profile server side |

Final JSON summary example:
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

---

## Typical Bisect Loop

1. Restore DB to pre‑test snapshot.  
2. Deploy/checkout the target Moodle code revision.  
3. Run the same load + Docker capture commands.  
4. Record p95/p99 latency, observed RPM, and container CPU/mem.  
5. Repeat to isolate the regression.

---

## Requirements

```
aiohttp>=3.9,<4.0
psycopg2-binary>=2.9
bcrypt>=4.0
docker>=7.0
```

---

## License

MIT License © 2025 Matt Porritt
