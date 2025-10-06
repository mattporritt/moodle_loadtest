# Moodle LMS Load Generator (async)

A Python toolkit for generating **repeatable load** on a Moodle LMS test site and diagnosing performance regressions.  
Includes:
- Async load testing with progress tables + CSV output  
- Docker resource capture with live container tables  
- End-to-end bisect automation support

---

## Repository Contents

| File | Description |
|------|--------------|
| **`moodle_load.py`** | Async load generator — logs in users, hits URLs, prints tables, writes CSV. |
| **`set_moodle_passwords.py`** | Resets test user passwords (Postgres) and populates `config.json`. |
| **`capture_docker_stats.py`** | Captures Docker CPU/mem/net I/O with live table and CSV output. |
| **`requirements.txt`** | Python dependencies. |

---

## Setup

### 1. Create and activate your environment

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

```bash
./bin/moodle-docker-compose exec webserver php public/admin/tool/generator/cli/maketestsite.php --size=M
```

### Step 2: Reset passwords and update config

```bash
python set_moodle_passwords.py   --db-name moodle   --db-user moodleuser   --db-pass S3cret   --db-host localhost   --count 200   --password Passw0rd!   --config config.json
```

### Step 3: Backup the database

```bash
pg_dump -U moodleuser -h localhost -F c -f pretest_backup.dump moodle
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
python moodle_load.py   --config config.json   --rpm 600   --duration 600   --concurrency 30   --stats-dir stats   --insecure
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
python capture_docker_stats.py   --containers moodlemaster-webserver-1 moodlemaster-db-1   --interval 1   --duration 600   --outdir stats   --tag $(git rev-parse --short HEAD)   --human   --print-interval 5
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

## 🧩 Combined Test Workflow (Recommended)

Run both load generation and Docker monitoring together for complete performance profiling.

```bash
# Clean up and prepare
rm -rf stats && mkdir stats

# Start Docker stats capture in the background
python capture_docker_stats.py   --containers moodlemaster-webserver-1 moodlemaster-db-1   --interval 1   --duration 600   --outdir stats   --tag $(git rev-parse --short HEAD)   --human   --print-interval 10 &

# Capture its PID
CAPTURE_PID=$!

# Run the load test
python moodle_load.py   --config config.json   --rpm 600   --duration 600   --concurrency 30   --stats-dir stats   --insecure

# Wait for Docker capture to finish
wait $CAPTURE_PID
```

This produces synchronized load and resource data sets:
```
stats/
 ├─ load_progress_<timestamp>.csv
 ├─ load_summary_<timestamp>.csv
 ├─ moodlemaster-webserver-1_<timestamp>_<tag>.csv
 └─ moodlemaster-db-1_<timestamp>_<tag>.csv
```

---

## Restoring Site to Pre-Test State

```bash
pg_restore -U moodleuser -h localhost -d moodle -c pretest_backup.dump
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

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program.  If not, see <http://www.gnu.org/licenses/>.