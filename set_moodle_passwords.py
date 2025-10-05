#!/usr/bin/env python3
"""
set_moodle_passwords.py — Prepare test users for load runs (PostgreSQL)

What it does
------------
- Connects directly to a Moodle PostgreSQL database.
- Selects a number of (non-deleted, non-suspended, id>2) users (most recent first).
- Sets their password to a *known* bcrypt value and forces `auth='manual'`.
- Writes the selected usernames + plaintext password into your load `config.json`.

Why direct DB updates?
----------------------
For performance test harnessing we often need a large pool of known credentials.
Doing this directly is fast and reproducible. **Run this only against test DBs.**

Security note
-------------
The load generator needs the plaintext password, so this script writes it into
`config.json` for convenience. Treat that file as sensitive and keep it out of
any production contexts or public repos.
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
    """Connection parameters for PostgreSQL."""
    host: str
    port: int
    dbname: str
    user: str
    password: str


def parse_args() -> argparse.Namespace:
    """CLI for selecting and updating test users and emitting config.json."""
    p = argparse.ArgumentParser(description="Set known passwords for Moodle users and populate load config.json")
    p.add_argument("--db-host", default="localhost", help="Postgres host")
    p.add_argument("--db-port", type=int, default=5432, help="Postgres port")
    p.add_argument("--db-name", required=True, help="Postgres database name")
    p.add_argument("--db-user", required=True, help="Postgres user")
    p.add_argument("--db-pass", required=True, help="Postgres password")
    p.add_argument("--prefix", default="mdl_", help="Moodle table prefix (default: mdl_)")
    p.add_argument("--count", type=int, default=100, help="How many users to update (default: 100)")
    p.add_argument("--password", required=True, help="New clear-text password for all selected users")
    p.add_argument("--config", default="config.json", help="Path to load generator config.json to update")
    p.add_argument("--where", default=None, help="Optional SQL filter (without leading WHERE), e.g. \"username LIKE 'perf%%'\"")
    p.add_argument("--dry-run", action="store_true", help="Print affected users but do not modify DB")
    p.add_argument("--bcrypt-cost", type=int, default=12, help="bcrypt cost factor (default: 12)")
    return p.parse_args()


def make_conn(cfg: DBConfig):
    """Open a psycopg2 connection."""
    return psycopg2.connect(
        host=cfg.host,
        port=cfg.port,
        dbname=cfg.dbname,
        user=cfg.user,
        password=cfg.password,
    )


def select_users(conn, prefix: str, count: int, where_clause: Optional[str]) -> List[dict]:
    """
    Return recent Moodle users (id desc) that are eligible for test logins.

    Baseline WHERE:
      - deleted = 0
      - id > 2 (skips admin/guest/system)
      - suspended = 0
    Extra filters can be provided via --where (no leading WHERE).
    """
    tbl = f"{prefix}user"
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
        return cur.fetchall()


def make_bcrypt_hash(plain: str, cost: int) -> bytes:
    """Generate a bcrypt hash compatible with PHP's password_verify()."""
    pw = plain.encode("utf-8")
    salt = bcrypt.gensalt(rounds=cost)
    return bcrypt.hashpw(pw, salt)  # bytes


def update_users(conn, prefix: str, user_rows: List[dict], bcrypt_hash: bytes, dry_run: bool = False) -> None:
    """
    Update each selected user's password hash and set auth='manual'.

    We also set confirmed=1 to avoid login prompts in some flows.
    """
    tbl = f"{prefix}user"
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
    """
    Merge/update the target config.json with a fresh 'users' list that pairs
    each selected username with the provided plaintext password (for the load generator).
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except FileNotFoundError:
        cfg = {}

    cfg.setdefault("base_url", cfg.get("base_url", "https://moodle.test"))
    cfg.setdefault("login_path", cfg.get("login_path", "/login/index.php"))

    cfg["users"] = [{"username": u["username"], "password": new_password} for u in users]

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"[CONFIG] Wrote {len(users)} users to {config_path}")


def main():
    args = parse_args()
    dbcfg = DBConfig(args.db_host, args.db_port, args.db_name, args.db_user, args.db_pass)
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

        bcrypt_hash = make_bcrypt_hash(args.password, args.bcrypt_cost)
        print(f"[INFO] Generated bcrypt hash with cost={args.bcrypt_cost} (length {len(bcrypt_hash)})")

        if args.dry_run:
            print("[DRY RUN] No DB changes will be made.")
        else:
            print("[INFO] Updating user passwords in DB...")
        update_users(conn, args.prefix, users, bcrypt_hash, dry_run=args.dry_run)

        update_config_json(args.config, users, args.password)
    finally:
        conn.close()


if __name__ == "__main__":
    main()