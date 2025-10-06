#!/usr/bin/env python3
"""
set_moodle_passwords.py
=======================
Utility to reset passwords for a subset of Moodle users and populate the
load generator configuration file with those credentials.

What it does
------------
- Connects directly to a PostgreSQL Moodle database
- Selects a set of user accounts (most recent first; excludes system users)
- Updates `mdl_user.password` with a bcrypt hash of your chosen password
- Sets `auth='manual'` and `confirmed=1` to ensure interactive login works
- Writes the selected usernames + the clear-text password into `config.json`

Safety
------
Run this only against non-production databases. It permanently changes user
passwords for the selected accounts.
"""
from __future__ import annotations

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
    """
    Connection parameters for PostgreSQL.
    """
    host: str
    port: int
    dbname: str
    user: str
    password: str


def parse_args() -> argparse.Namespace:
    """
    Define and parse all CLI flags for this utility.
    """
    p = argparse.ArgumentParser(
        description="Set known passwords for Moodle users and populate load config.json"
    )
    p.add_argument("--db-host", default="localhost", help="Postgres host")
    p.add_argument("--db-port", type=int, default=5432, help="Postgres port")
    p.add_argument("--db-name", required=True, help="Postgres database name")
    p.add_argument("--db-user", required=True, help="Postgres user")
    p.add_argument("--db-pass", required=True, help="Postgres password")
    p.add_argument("--prefix", default="mdl_", help="Moodle table prefix (default: mdl_)")
    p.add_argument("--count", type=int, default=100, help="Number of users to update (default: 100)")
    p.add_argument("--password", required=True, help="New clear-text password for selected users")
    p.add_argument("--config", default="config.json", help="Path to load generator config.json")
    p.add_argument(
        "--where",
        default=None,
        help="Optional SQL filter expression (without leading WHERE). "
             "Example: \"email LIKE 'test+%%' OR username LIKE 'perf%%'\"",
    )
    p.add_argument("--dry-run", action="store_true", help="Do not modify DB; only print and update config.json")
    p.add_argument("--bcrypt-cost", type=int, default=12, help="bcrypt cost factor (default: 12)")
    return p.parse_args()


def make_conn(cfg: DBConfig):
    """
    Open a PostgreSQL connection using plain psycopg2.

    Returns a live connection; caller is responsible for closing it.
    """
    return psycopg2.connect(
        host=cfg.host,
        port=cfg.port,
        dbname=cfg.dbname,
        user=cfg.user,
        password=cfg.password,
    )


def select_users(conn, prefix: str, count: int, where_clause: Optional[str]) -> List[dict]:
    """
    Select candidate users to update.

    Default filter excludes deleted/suspended/system users:
      - deleted = 0
      - suspended = 0
      - id > 2 (skip guest/admin)

    The selection is ordered by most recent (id DESC). You may append additional
    conditions using --where (no leading WHERE).
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
        return cur.fetchall()  # each row is a dict


def make_bcrypt_hash(plain: str, cost: int) -> bytes:
    """
    Create a bcrypt hash bytes for the given clear-text password.

    Moodle (PHP) uses password_hash/password_verify behind the scenes; bcrypt
    outputs are compatible.
    """
    pw = plain.encode("utf-8")
    salt = bcrypt.gensalt(rounds=cost)
    return bcrypt.hashpw(pw, salt)  # bytes


def update_users(conn, prefix: str, user_rows: List[dict], bcrypt_hash: bytes, dry_run: bool = False) -> None:
    """
    Apply the new hash to the selected users.

    We also set `auth='manual'` (to ensure user-pass login works) and
    `confirmed=1` (some flows check for confirmed users).
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
    Overwrite or create the users block in config.json with selected users.

    We leave other keys (base_url, login_path, parameters, urls) as-is.
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


def main() -> None:
    """
    Entry point for CLI execution.
    """
    args = parse_args()
    dbcfg = DBConfig(
        host=args.db_host, port=args.db_port, dbname=args.db_name, user=args.db_user, password=args.db_pass
    )

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

        # Finally reflect the user list + clear-text password into config.json
        update_config_json(args.config, users, args.password)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
