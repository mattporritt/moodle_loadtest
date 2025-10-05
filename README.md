# Moodle LMS Load Generator (async)

A Python toolkit for generating consistent, reproducible load on a Moodle LMS test site.  
It supports benchmarking different Moodle code versions (e.g. via `git bisect`), measuring performance, and restoring the site to a clean pre-test state.

---

## Overview

This repository provides two primary Python scripts:

1. **`moodle_load.py`** — Async load testing script that logs in test users and performs concurrent requests.
2. **`set_moodle_passwords.py`** — Utility for connecting to a PostgreSQL Moodle database, resetting test user passwords to a known value, and populating the load test configuration.

---

## Features

### Load Generator (`moodle_load.py`)
- Logs in multiple users (handles Moodle `logintoken`)
- Async parallel HTTP requests via `aiohttp`
- Target RPM (requests per minute) and duration control
- Randomised parameters in URLs (e.g. `{courseid}` → random int between range)
- Reports RPM, success/failure count, and latency percentiles (p50/p95/p99)

### Password Reset (`set_moodle_passwords.py`)
- Connects directly to PostgreSQL
- Selects test users (most recent or filtered with SQL)
- Updates passwords (bcrypt) and sets `auth='manual'`
- Updates `config.json` with usernames and passwords for load testing

---

## Preparing the Moodle Test Site

### Step 1: Generate test data

Moodle provides a built-in command-line generator for seeding test users and data:

```bash
./bin/moodle-docker-compose exec webserver php public/admin/tool/generator/cli/maketestsite.php --size=M
```

This creates users, courses, activities, and forum discussions; enough to simulate typical LMS load.

### Step 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate     # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Step 3: Reset test user passwords

Use the password script to assign a known password and populate the `config.json` file:

```bash
python3 set_moodle_passwords.py \
  --db-name moodle \
  --db-user moodleuser \
  --db-pass S3cret \
  --db-host localhost \
  --db-port 5433 \
  --prefix m_ \
  --count 50 \
  --password Passw0rd! \
  --config config.json
```

This will:
- Update `mdl_user.password` to a bcrypt hash of `Passw0rd!`
- Set `auth='manual'` and `confirmed=1`
- Write the selected users and password to `config.json`

### Step 4: Backup database before testing

```bash
pg_dump -U moodle -h 127.0.0.1 -p 5433 -Fc -f pretest_backup.dump moodle
```

### Step 5: Run the load test

```bash
python3 moodle_load.py \
  --config config.json \
  --rpm 600 \
  --duration 600 \
  --concurrency 30 \
  --insecure
```

### Step 6: Restore site to pre-test state

```bash
pg_restore -U moodleuser -h localhost -d moodle -c pretest_backup.dump
```

---

## Configuration File (`config.json`)

Example:

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

---

## Command Reference

### `moodle_load.py`

```
--config PATH          Path to config.json (required)
--rpm INT              Requests per minute target (required)
--duration INT         Duration in seconds (required)
--concurrency INT      Parallel workers (default: 20)
--insecure             Ignore TLS verification (self-signed/local)
--login-timeout INT    Login HTTP timeout seconds (default: 20)
--progress INT         Progress print interval seconds (default: 30)
```

### `set_moodle_passwords.py`

```
--db-name NAME         Database name (required)
--db-user USER         Database user (required)
--db-pass PASS         Database password (required)
--db-host HOST         Database host (default: localhost)
--db-port INT          Database port (default: 5432)
--prefix STR           Table prefix (default: mdl_)
--count INT            Number of users to update (default: 100)
--password STR         New password for selected users (required)
--config PATH          Path to load generator config.json (default: config.json)
--where SQL            Optional SQL filter (no leading WHERE)
--dry-run              Preview changes without modifying DB
```

---

## Understanding Output & Errors

| Message | Meaning |
|----------|----------|
| `[INFO] Logging in X users…` | Starting user authentication |
| `[ERROR] Login failed for user` | Invalid credentials or mismatched auth method |
| `[WARN] No logintoken for user` | Missing token – possible theme/custom login issue |
| `[Request error: ...]` | Network or timeout issue during HTTP GET |
| `[PROGRESS] {...}` | Current stats snapshot (RPM, latency) |
| `req_per_min_observed` | Achieved request rate (RPM) |
| `latency_ms_p95` | 95th percentile latency (ms) |

**Common fixes:**
- Increase `--concurrency` if observed RPM < target.
- Check that test users have course enrolments and permissions.
- Verify site URL and login path are correct.
- If many 403/404s occur, ensure URLs are valid for selected users.

---

## Ending the Test

Deactivate the virtual environment when finished:

```bash
deactivate
```

---

## Git Bisect Workflow

1. Establish baseline performance metrics.  
2. Run `git bisect start` and mark good/bad commits.  
3. Rebuild and restore the DB at each step.  
4. Run the same test, record p95/p99 latency and observed RPM.  
5. Narrow to the commit introducing the regression.

---

## Requirements

```
aiohttp>=3.9,<4.0
psycopg2-binary>=2.9
bcrypt>=4.0
```

---
