#!/usr/bin/env python3
"""
capture_docker_stats.py

Capture CPU% and memory usage for one or more Docker containers at a fixed interval,
writing one CSV per container. Useful for pairing with load tests / git bisect runs.

Requirements:
  pip install docker

Example:
  python capture_docker_stats.py \    --containers moodlemaster-webserver-1 moodlemaster-db-1 \    --interval 1 --duration 600 --outdir stats --tag $(git rev-parse --short HEAD)

Output files (created under --outdir):
  stats/<container>_<YYYYmmdd-HHMMSS>_<tag>.csv
"""
import argparse
import csv
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import docker

def human_bytes(n: Optional[int]) -> str:
    if n is None:
        return ""
    units = ["B","KiB","MiB","GiB","TiB"]
    x = float(n)
    for u in units:
        if x < 1024 or u == units[-1]:
            return f"{x:.2f} {u}"
        x /= 1024

def cpu_percent(prev: Optional[dict], cur: dict) -> float:
    try:
        if prev is None:
            return 0.0
        cpu_delta = (
            cur["cpu_stats"]["cpu_usage"]["total_usage"]
            - prev["cpu_stats"]["cpu_usage"]["total_usage"]
        )
        sys_delta = (
            cur["cpu_stats"]["system_cpu_usage"]
            - prev["cpu_stats"]["system_cpu_usage"]
        )
        cpus = (
            cur["cpu_stats"].get("online_cpus")
            or len(cur["cpu_stats"]["cpu_usage"].get("percpu_usage", []))
            or 1
        )
        if cpu_delta > 0 and sys_delta > 0:
            return (cpu_delta / sys_delta) * cpus * 100.0
        return 0.0
    except Exception:
        return 0.0

def net_bytes(cur: dict) -> Tuple[Optional[int], Optional[int]]:
    nets = cur.get("networks") or {}
    rx = sum(v.get("rx_bytes", 0) for v in nets.values()) if isinstance(nets, dict) else None
    tx = sum(v.get("tx_bytes", 0) for v in nets.values()) if isinstance(nets, dict) else None
    return rx, tx

def blk_bytes(cur: dict) -> Tuple[Optional[int], Optional[int]]:
    try:
        bio = cur.get("blkio_stats", {}) or {}
        read = 0
        write = 0
        for entry in bio.get("io_service_bytes_recursive", []) or []:
            if entry.get("op") == "Read":
                read += int(entry.get("value", 0))
            elif entry.get("op") == "Write":
                write += int(entry.get("value", 0))
        return read, write
    except Exception:
        return None, None

def main():
    ap = argparse.ArgumentParser(description="Capture docker stats to CSV for selected containers.")
    ap.add_argument("--containers", nargs="+", required=True, help="Container names or IDs (space-separated)")
    ap.add_argument("--interval", type=float, default=1.0, help="Sampling interval in seconds (default: 1)")
    ap.add_argument("--duration", type=int, default=600, help="Total duration in seconds (default: 600)")
    ap.add_argument("--outdir", default="stats", help="Output directory for CSV files (default: stats)")
    ap.add_argument("--tag", default="", help="Optional tag to include in filenames (e.g., git hash)")
    ap.add_argument("--human", action="store_true", help="Also write human-readable columns for memory and IO")
    args = ap.parse_args()

    client = docker.from_env()

    targets = {}
    for name in args.containers:
        try:
            targets[name] = client.containers.get(name)
        except Exception as e:
            print(f"[ERROR] Cannot find container '{name}': {e}", file=sys.stderr)
    if not targets:
        print("[FATAL] No valid containers found.", file=sys.stderr)
        sys.exit(2)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ts_str = datetime.now().strftime("%Y%m%d-%H%M%S")

    writers: Dict[str, csv.writer] = {}
    files = {}
    prev_stats: Dict[str, Optional[dict]] = {k: None for k in targets.keys()}

    suffix = f"_{args.tag}" if args.tag else ""

    headers = [
        "timestamp",
        "container",
        "cpu_percent",
        "mem_used_bytes",
        "mem_limit_bytes",
        "pids",
        "net_rx_bytes",
        "net_tx_bytes",
        "blk_read_bytes",
        "blk_write_bytes",
    ]
    if args.human:
        headers += [
            "mem_used_human",
            "mem_limit_human",
            "net_rx_human",
            "net_tx_human",
            "blk_read_human",
            "blk_write_human",
        ]

    for name in targets.keys():
        path = outdir / f"{name}_{ts_str}{suffix}.csv"
        f = open(path, "w", newline="")
        files[name] = f
        w = csv.writer(f)
        w.writerow(headers)
        writers[name] = w
        print(f"[INFO] Writing: {path}")

    stop = False
    def handle_sig(signum, frame):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    start = time.time()
    next_tick = start
    while not stop and (time.time() - start) < args.duration:
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%S")
        for name, ctr in targets.items():
            try:
                s = ctr.stats(stream=False)
            except Exception as e:
                print(f"[WARN] Failed to get stats for {name}: {e}", file=sys.stderr)
                continue

            cpu = cpu_percent(prev_stats[name], s)
            mem_used = s.get("memory_stats", {}).get("usage")
            mem_lim  = s.get("memory_stats", {}).get("limit")
            pids     = (s.get("pids_stats") or {}).get("current")
            rx, tx   = net_bytes(s)
            rb, wb   = blk_bytes(s)

            row = [
                now_iso, name, f"{cpu:.2f}", mem_used, mem_lim, pids, rx, tx, rb, wb
            ]
            if args.human:
                row += [
                    human_bytes(mem_used),
                    human_bytes(mem_lim),
                    human_bytes(rx),
                    human_bytes(tx),
                    human_bytes(rb),
                    human_bytes(wb),
                ]

            writers[name].writerow(row)
            prev_stats[name] = s

        next_tick += args.interval
        sleep_for = max(0.0, next_tick - time.time())
        time.sleep(sleep_for)

    for f in files.values():
        f.close()
    print("[INFO] Done.")

if __name__ == "__main__":
    main()
