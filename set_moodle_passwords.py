#!/usr/bin/env python3
"""set_moodle_passwords.py

Connects directly to a PostgreSQL Moodle database, selects a number of recent users,
sets their password to a known value (bcrypt) and writes their usernames + password
into an existing JSON config file (config.json) for the load generator.

CAVEATS:
- This script updates the Moodle user table directly. Run against a test DB only.
- It sets the `password` field to a bcrypt hash and `auth` to 'manual' so Moodle accepts the password.
  This should be compatible with modern PHP password_verify usage in Moodle.
- By default it picks the most recent users (id desc) excluding system users (id <= 2).
  Use --where to provide a custom SQL filter (without the leading WHERE).
"""

import argparse
import json
import sys
from dataclasses import dataclass
from typing import List, Optional

import bcrypt
import psycopg2
from psycopg2.extras import RealDictCursor


@dataclass
class DBConfig:
    host: str
    port: int
    dbname: str
    user: str
    password: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Set known passwords for Moodle users and populate load config.json")
    p.add_argument("--db-host", default="localhost", help="Postgres host")
    p.add_argument("--db-port", type=int, default=5432, help="Postgres port")
    p.add_argument("--db-name", required=True, help="Postgres database name")
    p.add_argument("--db-user", required=True, help="Postgres user")
    p.add_argument("--db-pass", required=True, help="Postgres password")
    p.add_argument("--prefix", default="mdl_", help="Table prefix used in Moodle (default: mdl_)")
    p.add_argument("--count", type=int, default=100, help="Number of users to find and update (default: 100)")
    p.add_argument("--password", required=True, help="New clear-text password to apply to all selected users")
    p.add_argument("--config", default="config.json", help="Path to load generator config.json to update (default: config.json)")
    p.add_argument("--where", default=None, help="Optional SQL filter expression for selecting users (e.g. \"email LIKE 'test+%%' or username LIKE 'perf%%'\") - DO NOT prepend WHERE")
    p.add_argument("--dry-run", action="store_true", help="Do not modify DB; just print which users would be updated and update config.json if requested")
    p.add_argument("--bcrypt-cost", type=int, default=12, help="bcrypt cost factor (default: 12)")
    return p.parse_args()


def make_conn(cfg: DBConfig):
    conn = psycopg2.connect(
        host=cfg.host,
        port=cfg.port,
        dbname=cfg.dbname,
        user=cfg.user,
        password=cfg.password,
    )
    return conn


def select_users(conn, prefix: str, count: int, where_clause: Optional[str]) -> List[dict]:
    tbl = f"{prefix}user"
    # Baseline filters: not deleted, id > 2 (avoid admin/system), not suspended
    where_parts = [f"{tbl}.deleted = 0", f"{tbl}.id > 2", f"{tbl}.suspended = 0"]
    if where_clause:
        where_parts.append(f"({where_clause})")
    where_sql = " AND ".join(where_parts)
    sql = f"""
    SELECT id, username, email, auth
    FROM {tbl}
    WHERE {where_sql}
    ORDER BY id DESC
    LIMIT %s
    """
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, (count,))
        rows = cur.fetchall()
    return rows


def make_bcrypt_hash(plain: str, cost: int) -> bytes:
    # bcrypt expects bytes input
    pw = plain.encode("utf-8")
    salt = bcrypt.gensalt(rounds=cost)
    h = bcrypt.hashpw(pw, salt)
    return h  # bytes


def update_users(conn, prefix: str, user_rows: List[dict], bcrypt_hash: bytes, dry_run: bool = False) -> None:
    tbl = f"{prefix}user"
    # We'll set: password = <hash>, auth = 'manual', confirmed = 1
    sql = f"""
    UPDATE {tbl}
    SET password = %s, auth = 'manual', confirmed = 1
    WHERE id = %s
    """
    with conn.cursor() as cur:
        for r in user_rows:
            uid = r["id"]
            uname = r["username"]
            if dry_run:
                print(f"[DRY RUN] Would update user id={uid} username={uname}")
            else:
                cur.execute(sql, (bcrypt_hash.decode("utf-8"), uid))
                print(f"[UPDATED] id={uid} username={uname}")
        if not dry_run:
            conn.commit()


def update_config_json(config_path: str, users: List[dict], new_password: str) -> None:
    # Read existing config or start new
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except FileNotFoundError:
        cfg = {}

    # Ensure base keys exist
    cfg.setdefault("base_url", cfg.get("base_url", "https://moodle.test"))
    cfg.setdefault("login_path", cfg.get("login_path", "/login/index.php"))

    # Replace users block
    cfg["users"] = [{"username": u["username"], "password": new_password} for u in users]

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"[CONFIG] Wrote {len(users)} users to {config_path}")


def main():
    args = parse_args()
    dbcfg = DBConfig(host=args.db_host, port=args.db_port, dbname=args.db_name, user=args.db_user, password=args.db_pass)
    try:
        conn = make_conn(dbcfg)
    except Exception as e:
        print(f"[ERROR] Could not connect to DB: {e}")
        sys.exit(1)

    try:
        users = select_users(conn, args.prefix, args.count, args.where)
        if not users:
            print("[INFO] No users matched the selection - nothing to do.")
            return

        print(f"[INFO] Selected {len(users)} users (most recent first):")
        for u in users:
            print(f" - id={u['id']} username={u['username']} email={u.get('email')} auth={u.get('auth')}")

        # generate bcrypt hash
        bcrypt_hash = make_bcrypt_hash(args.password, args.bcrypt_cost)
        print(f"[INFO] Generated bcrypt hash with cost={args.bcrypt_cost} (length {len(bcrypt_hash)})")

        if args.dry_run:
            print("[DRY RUN] No DB changes will be made.")
        else:
            print("[INFO] Updating user passwords in DB...")
        update_users(conn, args.prefix, users, bcrypt_hash, dry_run=args.dry_run)

        # update config.json to include these users and the plaintext password for the loadgen
        update_config_json(args.config, users, args.password)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
