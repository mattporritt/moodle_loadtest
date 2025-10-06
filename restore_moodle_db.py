#!/usr/bin/env python3
"""
restore_moodle_db.py
====================
Restore a PostgreSQL Moodle database from a `pg_dump` (custom format) file.

This script is designed to support repeatable performance testing workflows
(e.g., `git bisect`) by providing a **clean, deterministic restore** between runs.

Why Python (not bash)?
----------------------
- Clear, structured error handling and logging
- Works consistently on macOS/Linux/Windows
- Reuses the same DB access pattern as the password reset script

Assumptions
-----------
- The dump file was created with `pg_dump -F c --no-owner --no-privileges`.
- The calling user has permissions to terminate connections and either:
  * drop/create the target DB (for `--mode drop`), or
  * drop/recreate objects within the DB (for `--mode clean`).

Two restore modes
-----------------
1) **clean** (default): keep the database, drop & recreate all objects from the dump
   - Equivalent to `pg_restore --clean --if-exists`
   - Avoids CREATE DATABASE permission issues

2) **drop**: drop and recreate the database, then restore all objects
   - Requires privileges to DROP/CREATE DATABASE
   - Useful if you want a “true” rollback

Examples
--------
# Clean restore (keep DB, drop/recreate objects)
python restore_moodle_db.py \\
  --db-name moodle --db-user moodleuser --db-pass S3cret --db-host localhost \\
  --dump pretest_backup.dump --mode clean --jobs 4

# Drop & recreate database before restore
python restore_moodle_db.py \\
  --db-name moodle --db-user moodleuser --db-pass S3cret --db-host localhost \\
  --dump pretest_backup.dump --mode drop --jobs 4
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT


@dataclass
class DBConfig:
    host: str
    port: int
    dbname: str
    user: str
    password: str


def _info(msg: str) -> None:
    print(f"[INFO] {msg}")


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def _error(msg: str) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)


def terminate_connections(cfg: DBConfig) -> None:
    sql = f"""
        SELECT pg_terminate_backend(pid)
        FROM pg_stat_activity
        WHERE datname = %s
          AND pid <> pg_backend_pid();
    """
    conn = psycopg2.connect(
        host=cfg.host, port=cfg.port, dbname="postgres", user=cfg.user, password=cfg.password
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (cfg.dbname,))
        _info("Terminated existing connections to target database.")
    finally:
        conn.close()


def drop_and_create_database(cfg: DBConfig) -> None:
    conn = psycopg2.connect(
        host=cfg.host, port=cfg.port, dbname="postgres", user=cfg.user, password=cfg.password
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    try:
        with conn.cursor() as cur:
            _info(f"Dropping database if exists: {cfg.dbname}")
            cur.execute(f"DROP DATABASE IF EXISTS {psql_ident(cfg.dbname)};")
            _info(f"Creating database: {cfg.dbname} (owner: {cfg.user})")
            cur.execute(
                f"CREATE DATABASE {psql_ident(cfg.dbname)} OWNER {psql_ident(cfg.user)} TEMPLATE template0;"
            )
    finally:
        conn.close()


def psql_ident(ident: str) -> str:
    if not ident.isidentifier() or ident.lower() != ident:
        return '"' + ident.replace('"', '""') + '"'
    return ident


def run_pg_restore(
    cfg: DBConfig,
    dump_path: Path,
    jobs: int,
    mode: str,
    pg_restore_path: Optional[str] = None,
) -> None:
    if pg_restore_path is None:
        pg_restore_path = "pg_restore"

    base_args = [
        pg_restore_path,
        "-U", cfg.user,
        "-h", cfg.host,
        "-d", cfg.dbname,
        "--no-owner",
        "--no-privileges",
        "-j", str(max(1, jobs)),
        str(dump_path),
    ]

    if mode == "clean":
        args = base_args[:]
        args.insert(-1, "--clean")
        args.insert(-1, "--if-exists")
    elif mode == "drop":
        args = base_args
    else:
        raise ValueError("mode must be 'clean' or 'drop'")

    _info("Running pg_restore: " + " ".join(shlex.quote(a) for a in args))
    env = os.environ.copy()
    env["PGPASSWORD"] = cfg.password

    try:
        subprocess.run(args, check=True, env=env)
    except subprocess.CalledProcessError as e:
        _error(f"pg_restore failed with exit code {e.returncode}")
        raise


def analyze_database(cfg: DBConfig) -> None:
    conn = psycopg2.connect(
        host=cfg.host, port=cfg.port, dbname=cfg.dbname, user=cfg.user, password=cfg.password
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    try:
        with conn.cursor() as cur:
            _info("Running ANALYZE (this may take a moment)...")
            cur.execute("ANALYZE;")
    finally:
        conn.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Restore a PostgreSQL Moodle database from a pg_dump (custom format) file."
    )
    parser.add_argument("--db-host", default="localhost", help="Postgres host (default: localhost)")
    parser.add_argument("--db-port", type=int, default=5432, help="Postgres port (default: 5432)")
    parser.add_argument("--db-name", required=True, help="Target database name (e.g., moodle)")
    parser.add_argument("--db-user", required=True, help="Database user (owner or superuser)")
    parser.add_argument("--db-pass", required=True, help="Database password for the user")
    parser.add_argument("--dump", required=True, help="Path to pg_dump custom-format file (*.dump)")
    parser.add_argument("--mode", choices=["clean", "drop"], default="clean", help="Restore mode (default: clean)")
    parser.add_argument("--jobs", type=int, default=4, help="Parallel jobs for pg_restore -j (default: 4)")
    parser.add_argument("--pg-restore-path", default=None, help="Optional path to pg_restore binary")
    parser.add_argument("--skip-analyze", action="store_true", help="Skip ANALYZE after restore")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dump_path = Path(args.dump)
    if not dump_path.exists():
        _error(f"Dump file does not exist: {dump_path}")
        sys.exit(2)

    cfg = DBConfig(
        host=args.db_host,
        port=args.db_port,
        dbname=args.db_name,
        user=args.db_user,
        password=args.db_pass,
    )

    terminate_connections(cfg)

    if args.mode == "drop":
        drop_and_create_database(cfg)
    else:
        _info("Proceeding with in-place clean restore (no dropdb).")

    run_pg_restore(cfg, dump_path, jobs=args.jobs, mode=args.mode, pg_restore_path=args.pg_restore_path)

    if not args.skip_analyze:
        try:
            analyze_database(cfg)
        except Exception as e:
            _warn(f"ANALYZE failed or timed out: {e}")

    _info("Restore complete.")


if __name__ == "__main__":
    main()
