#!/usr/bin/env python3
"""
capture_docker_stats.py
=======================
Capture CPU%, memory usage, PIDs, network bytes, and block I/O for one or more
Docker containers at a fixed interval. Writes one CSV per container and prints
a periodic, human-readable summary table to the terminal.

Why
---
Pair this script with the load generator to correlate application-level metrics
(latency/RPM) with system resource usage of your Moodle containers.

Example
-------
python capture_docker_stats.py \
  --containers moodlemaster-webserver-1 moodlemaster-db-1 \
  --interval 1 --duration 600 --outdir stats --tag $(git rev-parse --short HEAD) \
  --human --print-interval 5
"""
from __future__ import annotations

import argparse
import csv
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import docker


# ---------- Formatting helpers ----------
def human_bytes(n: Optional[int]) -> str:
    """
    Render a byte count into a human-readable string (KiB, MiB, GiB...).

    Returns an empty string for None so table rendering stays clean.
    """
    if n is None:
        return ""
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    x = float(n)
    for u in units:
        if x < 1024 or u == units[-1]:
            return f"{x:.2f} {u}"
        x /= 1024


def _format_table(headers, rows):
    """
    Simple 2D table output for the terminal (see moodle_load._format_table).
    """
    cols = len(headers)
    widths = [len(h) for h in headers]
    for row in rows:
        for i in range(cols):
            widths[i] = max(widths[i], len(str(row[i]) if i < len(row) else ""))
    sep = "+".join("-" * (w + 2) for w in widths)
    lines = []
    header_line = " | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers))
    lines.append(header_line)
    lines.append(sep)
    for row in rows:
        line = " | ".join(str(row[i]).ljust(widths[i]) for i in range(cols))
        lines.append(line)
    return "\n".join(lines)


# ---------- Metric extraction ----------
def cpu_percent(prev: Optional[dict], cur: dict) -> float:
    """
    Compute CPU percentage using Docker's cumulative counters.

    Formula mirrors the Docker CLI:
        cpu% = (cpu_delta / sys_delta) * online_cpus * 100
    where deltas are between consecutive samples.
    """
    try:
        if prev is None:
            return 0.0
        cpu_delta = cur["cpu_stats"]["cpu_usage"]["total_usage"] - prev["cpu_stats"]["cpu_usage"]["total_usage"]
        sys_delta = cur["cpu_stats"]["system_cpu_usage"] - prev["cpu_stats"]["system_cpu_usage"]
        cpus = cur["cpu_stats"].get("online_cpus") or len(cur["cpu_stats"]["cpu_usage"].get("percpu_usage", [])) or 1
        if cpu_delta > 0 and sys_delta > 0:
            return (cpu_delta / sys_delta) * cpus * 100.0
        return 0.0
    except Exception:
        return 0.0


def net_bytes(cur: dict) -> Tuple[Optional[int], Optional[int]]:
    """
    Sum total rx/tx across all container network interfaces.

    Fields may be missing depending on driver; we defensively default to 0.
    """
    nets = cur.get("networks") or {}
    rx = sum(v.get("rx_bytes", 0) for v in nets.values()) if isinstance(nets, dict) else None
    tx = sum(v.get("tx_bytes", 0) for v in nets.values()) if isinstance(nets, dict) else None
    return rx, tx


def blk_bytes(cur: dict) -> Tuple[Optional[int], Optional[int]]:
    """
    Sum block I/O read/write bytes from Docker stats.

    Not all storage drivers expose blkio stats; handle missing keys gracefully.
    """
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


# ---------- Orchestration ----------
def main() -> None:
    """
    Capture loop: open CSV per container, poll stats at interval, print table snapshots.
    """
    ap = argparse.ArgumentParser(description="Capture docker stats to CSV for selected containers, with table output.")
    ap.add_argument("--containers", nargs="+", required=True, help="Container names or IDs")
    ap.add_argument("--interval", type=float, default=1.0, help="Sampling interval seconds (default: 1)")
    ap.add_argument("--duration", type=int, default=600, help="Total duration seconds (default: 600)")
    ap.add_argument("--outdir", default="stats", help="Directory for output CSV files (default: stats)")
    ap.add_argument("--tag", default="", help="Optional tag to include in filenames (e.g., git hash)")
    ap.add_argument("--human", action="store_true", help="Append human-readable columns to CSV")
    ap.add_argument("--print-interval", type=int, default=5, help="How often (seconds) to print a table snapshot")
    args = ap.parse_args()

    client = docker.from_env()

    # Resolve containers eagerly to fail fast if names are wrong.
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
    suffix = f"_{args.tag}" if args.tag else ""

    # Prepare CSV writers and previous-sample holders
    writers: Dict[str, csv.writer] = {}
    files = {}
    prev_stats: Dict[str, Optional[dict]] = {k: None for k in targets.keys()}
    last_print = time.time()

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

    # Graceful termination on signals (Ctrl-C)
    stop = False
    def handle_sig(signum, frame):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    # Main sampling loop
    start = time.time()
    next_tick = start
    latest_rows: Dict[str, Dict[str, str]] = {}

    while not stop and (time.time() - start) < args.duration:
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%S")

        # Poll each container once per loop
        for name, ctr in targets.items():
            try:
                s = ctr.stats(stream=False)
            except Exception as e:
                print(f"[WARN] Failed to get stats for {name}: {e}", file=sys.stderr)
                continue

            # Compute/collect metrics
            cpu = cpu_percent(prev_stats[name], s)
            mem_used = s.get("memory_stats", {}).get("usage")
            mem_lim = s.get("memory_stats", {}).get("limit")
            pids = (s.get("pids_stats") or {}).get("current")
            rx, tx = net_bytes(s)
            rb, wb = blk_bytes(s)

            # Write raw metrics to CSV
            row = [now_iso, name, f"{cpu:.2f}", mem_used, mem_lim, pids, rx, tx, rb, wb]
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
            files[name].flush()
            prev_stats[name] = s

            # Keep a friendly copy for terminal table
            latest_rows[name] = {
                "container": name,
                "cpu": f"{cpu:.2f}%",
                "mem": f"{human_bytes(mem_used)}/{human_bytes(mem_lim)}" if args.human else f"{mem_used}/{mem_lim}",
                "pids": str(pids or ""),
                "net": f"rx {human_bytes(rx)} tx {human_bytes(tx)}" if args.human else f"rx {rx} tx {tx}",
                "blk": f"r {human_bytes(rb)} w {human_bytes(wb)}" if args.human else f"r {rb} w {wb}",
            }

        # Print a compact table every --print-interval seconds
        if (time.time() - last_print) >= args.print_interval:
            headers_tbl = ["Container", "CPU%", "Memory (used/limit)", "PIDs", "Net (rx/tx)", "Block I/O (r/w)"]
            rows_tbl = []
            for cname in sorted(latest_rows.keys()):
                r = latest_rows[cname]
                rows_tbl.append([r["container"], r["cpu"], r["mem"], r["pids"], r["net"], r["blk"]])
            if rows_tbl:
                print("\n[CONTAINER STATS]\n" + _format_table(headers_tbl, rows_tbl))
            last_print = time.time()

        # Sleep to maintain a stable sampling interval
        next_tick += args.interval
        time.sleep(max(0.0, next_tick - time.time()))

    # Flush/close files
    for f in files.values():
        f.close()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
