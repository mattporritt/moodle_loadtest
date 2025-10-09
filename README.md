# Moodle LMS Load Generator (async)

A Python toolkit for generating **repeatable load** on a Moodle LMS test site and diagnosing performance regressions.  
Includes:
- Async load testing with progress tables + CSV output  
- Docker resource capture with live container tables  
- PostgreSQL **database restore** utility for consistent, per-run resets  
- End-to-end bisect automation support

---

## Repository Contents

| File | Description                                                                                    |
|------|------------------------------------------------------------------------------------------------|
| **`moodle_load.py`** | Async load generator, logs in users, hits URLs, prints tables, writes CSV.                     |
| **`set_moodle_passwords.py`** | Resets test user passwords (Postgres) and populates `config.json`.                             |
| **`capture_docker_stats.py`** | Captures Docker CPU/mem/net I/O with live table and CSV output.                                |
| **`restore_moodle_db.py`** | **New:** Restores the Moodle Postgres DB from a `pg_dump -F c` snapshot (clean or drop modes). |
| **`requirements.txt`** | Python dependencies.                                                                           |

---

## Setup

### 1. Create and activate your environment
Once cloning the code, you'll need to activate the Python virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

To deactivate later:
```bash
deactivate
```

---

## Preparing the Moodle Test Site

### Step 1: Seed test data
Moodle LMS has an inbuilt tool to generate test data.    
This creates a site with a realistic number of users, courses, and activities.   
Use this script to generate a site of a given size for load testing.

Example: On webserver directly:
```bash
php public/admin/tool/generator/cli/maketestsite.php --size=M
```

Example: Docker based webserver:
```bash
./bin/moodle-docker-compose exec webserver php public/admin/tool/generator/cli/maketestsite.php --size=M
```

### Step 2 (OPTIONAL): Add extra enrolments)
The site generator creates a number of users and courses, but not all users are enrolled in many courses. Also, all users are enrolled as students only, there are no teachers.   
If you generated the site with a medium size (`--size=M`), 1,000 users would have been created along with 72 courses.
This repository contains a `users.csv` file that can be optionally used to increase user enrolments.  
The following commands will enrol all users in all courses, mostly as students. There will be 10 courses where users are enrolled as teachers.   

**NOTE:** The `users.csv` file will need to be accessible to the webserver, so if using Docker, copy it to the webserver container first.   
For example:
```bash
cp -v ~/moodle_loadtest/users.csv ./
```

Use the Moodle LMS commandline tool to process the CSV and enrol users.   
On webserver directly:
```bash
php admin/tool/uploaduser/cli/uploaduser.php —file=/var/www/html/Users.csv --uutype=3
```

Example: Docker based webserver:
```bash
./bin/moodle-docker-compose exec webserver php admin/tool/uploaduser/cli/uploaduser.php —file=/var/www/html/Users.csv --uutype=3
```

### Step 3: Reset passwords and update config
The site generator creates users with random passwords.   
To allow the load testing script to log in, we need to reset a number of user passwords to a known value and update `config.json` with the usernames.
Use the `set_moodle_passwords.py` script to do this:

```bash
python set_moodle_passwords.py \
  --db-name moodle \
  --db-user moodleuser \
  --db-pass S3cret \
  --db-host 127.0.0.1 \
  --db-port 5433 \
  --prefix m_ \
  --count 1000 \
  --password Passw0rd! \
  --config config.json
```

### Step 4: Backup the database (pre-test snapshot)
In order to get consistent results between test runs, it's important to reset the database to a known state before each run.    
Use `pg_dump` to create a snapshot of the database after seeding and password reset so restores are fast and portable. 

To use `pg_dump`, you will need to have the Postgres client tools installed.   
The following example creates a custom-format dump file without ownership or privilege commands, which is suitable for use with the `restore_moodle_db.py` script.

```bash
pg_dump -U moodle \
    -h localhost \
    -p 5433 \
    -F c -f pretest_backup.dump \
    --no-owner \
    --no-privileges moodle
```

---

## Load Test Configuration (`config.json`)

```json
{
  "base_url": "https://moodle.test",
  "login_path": "/login/index.php",
  "users": [
    { "username": "perfuser01", "password": "Passw0rd!" }
  ],
  "parameters": {
    "courseid": { "min": 2, "max": 500 },
    "userid": { "min": 5, "max": 10000 }
  },
  "urls": [
    { "path": "/my/" },
    { "path": "/course/view.php?id={courseid}" }
  ]
}
```

---

## Running the Load Test

```bash
python moodle_load.py \
    --config config.json \
    --rpm 600 \
    --duration 600 \
    --concurrency 30 \
    --stats-dir stats
```

### Output example

```
[PROGRESS]
Metric         | Value
------------------------
Elapsed (s)    | 300.0
Target RPM     | 600
Observed RPM   | 598.5
Total          | 3000
Success        | 2992
Failures       | 8
p50 (ms)       | 120
p95 (ms)       | 350
p99 (ms)       | 590
```

At completion, a final table and JSON summary are printed and CSVs are written to:
```
stats/load_progress_<timestamp>.csv
stats/load_summary_<timestamp>.csv
```

---

## Monitoring Docker Containers

```bash
python capture_docker_stats.py \
    --containers moodlemaster-webserver-1 moodlemaster-db-1 \
    --interval 1 \
    --duration 600 \
    --outdir stats \
    --tag $(git rev-parse --short HEAD) \
    --human \
    --print-interval 5
```

### Console table output

```
[CONTAINER STATS]
Container                | CPU%  | Memory (used/limit) | PIDs | Net (rx/tx)             | Block I/O (r/w)
-----------------------------------------------------------------------------------------------------------
moodlemaster-webserver-1 | 42.3% | 450.00 MiB/2.00 GiB |  18  | rx 10.2 MiB tx 7.8 MiB  | r 98 MiB w 4 MiB
moodlemaster-db-1        | 66.7% | 930.00 MiB/2.00 GiB |  15  | rx 2.2 MiB tx 1.7 MiB   | r 320 MiB w 75 MiB
```

### Files written
```
stats/moodlemaster-webserver-1_<timestamp>_<tag>.csv
stats/moodlemaster-db-1_<timestamp>_<tag>.csv
```

---

## Database Restore Between Runs (New)

Use **`restore_moodle_db.py`** to reset the database to your pre-test snapshot.

> Assumes the dump was created with:  
> `pg_dump -F c --no-owner --no-privileges`

### Clean restore (default; no dropdb)
Keeps the database, drops/recreates all objects from the dump:
```bash
python restore_moodle_db.py \
  --db-name moodle \
  --db-user moodleuser \
  --db-pass S3cret \
  --db-host localhost \
  --db-port 5433 \
  --dump pretest_backup.dump \
  --mode clean \
  --jobs 4
```

### Drop & recreate database
Preferred when you want a full DB reset (requires DROP/CREATE privileges):
```bash
python restore_moodle_db.py \
  --db-name moodle \
  --db-user moodleuser \
  --db-pass S3cret \
  --db-host localhost \
  --dump pretest_backup.dump \
  --mode drop \
  --jobs 4
```

**Options**
- `--jobs N` : Parallel workers for `pg_restore -j` (defaults to 4).  
- `--pg-restore-path PATH` : Use a specific `pg_restore` binary (optional).  
- `--skip-analyze` : Skip the final `ANALYZE;` step.

**What the restore script does**
1. Terminates active sessions to the target DB (avoids lock errors).  
2. Either:  
   - **clean**: in-place restore with `pg_restore --clean --if-exists`, or  
   - **drop**: `DROP DATABASE` / `CREATE DATABASE` then restore.  
3. Runs `ANALYZE;` to refresh planner stats (can be disabled).

**Docker note:** If Postgres runs in Docker without a published port, either publish `5432` or `docker exec` the `pg_dump`/`pg_restore` commands inside the container. The Python restore script can run anywhere it can reach the DB host.

---

## Connection and Concurrency Controls

To prevent overloading your test system or the Moodle server, you can fine-tune how connections and logins are handled.

### `--login-concurrency`
- **Purpose:** Limits how many users attempt to log in simultaneously.  
- **Default:** `20`  
- **Effect:** Reduces initial load on the web server and DB during the login phase.

### `--connector-limit` and `--connector-limit-per-host`
- **Purpose:** Control how many network connections each simulated user can open at once.  
- **Defaults:** `8` (total) and `4` (per host)  
- **Effect:** Prevents socket exhaustion and “Too many open files” errors.

### Example usage
```bash
python moodle_load.py \
  --config config.json \
  --rpm 600 \
  --duration 600 \
  --concurrency 30 \
  --login-concurrency 20 \
  --connector-limit 8 \
  --connector-limit-per-host 4
```

| Setting | Prevents | Typical safe value |
|----------|-----------|--------------------|
| `--login-concurrency` | Login storm (server overload at start) | 20 |
| `--connector-limit` | File descriptor exhaustion (too many sockets) | 8 |
| `--connector-limit-per-host` | Too many connections to one host | 4 |

---

## Combined Test Workflow (Recommended)

Run both load generation and Docker monitoring in parallel; reset DB between runs using the restore script.

```bash
# Clean up and prepare
rm -rf stats && mkdir stats

# Start Docker stats capture in the background
python capture_docker_stats.py \
  --containers moodlemaster-webserver-1 moodlemaster-db-1 \
  --interval 1 \
  --duration 600 \
  --outdir stats \
  --tag $(git rev-parse --short HEAD) \
  --human \
  --print-interval 10 &

CAPTURE_PID=$!

# Run the load test
python moodle_load.py \
  --config config.json \
  --rpm 600 \
  --duration 600 \
  --concurrency 30 \
  --stats-dir stats \
  --insecure

# Wait for Docker capture to finish
wait $CAPTURE_PID

# Reset DB to snapshot for next run
python restore_moodle_db.py \
  --db-name moodle \
  --db-user moodleuser \
  --db-pass S3cret \
  --db-host localhost \
  --dump pretest_backup.dump \
  --mode clean \
  --jobs 4
```

---

## Data Interpretation

| Metric | Description |
|---------|--------------|
| **Observed RPM** | Real achieved request rate vs. target |
| **p95 / p99 latency** | Tail latency of Moodle responses |
| **CPU% / Memory** | Container resource utilization |
| **Block I/O** | Disk read/write throughput (DB-heavy clues) |

**Tips for analysis:**
- Large latency increases with stable CPU → likely code-path regression  
- High CPU/memory in DB → inefficient queries  
- Drop in observed RPM → request throttling or slow PHP execution  

---

### (Optional) Raise OS open-file limit temporarily

If you see `Too many open files` during heavy tests, increase the per-process
file descriptor limit **for the current shell** before running the load script:

**Linux/macOS (temporary for this shell):**
```bash
ulimit -n 65535
```

> Keep connector limits modest and throttle logins (defaults are safe).  
> For large runs, try: `--login-concurrency 20 --connector-limit 8 --connector-limit-per-host 4`.

---

## Requirements

```
aiohttp>=3.9,<4.0
psycopg2-binary>=2.9
bcrypt>=4.0
docker>=7.0
```

---

## License ##

2025 Matt Porritt <matt.porritt@moodle.com>

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.
