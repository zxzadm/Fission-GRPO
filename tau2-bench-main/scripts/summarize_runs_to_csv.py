#!/usr/bin/env python3
import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


TARGET_DOMAINS = ["retail", "airline", "telecom"]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SIM_DIR = PROJECT_ROOT / "data" / "simulations"
DEFAULT_OUTPUT = DEFAULT_SIM_DIR / "summary_by_time.csv"


def _to_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def parse_file_meta(path: Path) -> dict[str, str | None]:
    """
    Expected file name format:
      qs_YYYYMMDD_HHMMSS_<domain>_<mode>.json
    """
    stem = path.stem
    match = re.match(
        r"^(?P<prefix>.+?)_(?P<timestamp>\d{8}_\d{6})_(?P<domain>[^_]+)_(?P<mode>[^_]+)$",
        stem,
    )
    if match:
        return {
            "timestamp": match.group("timestamp"),
            "domain_from_name": match.group("domain"),
            "mode": match.group("mode"),
        }
    return {
        "timestamp": None,
        "domain_from_name": None,
        "mode": None,
    }


def summarize_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    meta = parse_file_meta(path)
    info = data.get("info", {})
    env_info = info.get("environment_info", {})
    agent_info = info.get("agent_info", {})
    user_info = info.get("user_info", {})

    domain = env_info.get("domain_name") or meta["domain_from_name"] or "unknown"
    simulations = data.get("simulations", []) or []
    tasks = data.get("tasks", []) or []

    rewards = []
    termination_counter = Counter()
    success_count = 0

    for sim in simulations:
        termination_counter[str(sim.get("termination_reason", "unknown"))] += 1
        reward = _to_float(sim.get("reward_info", {}).get("reward"))
        if reward is not None:
            rewards.append(reward)
            if reward >= 1.0:
                success_count += 1

    sim_count = len(simulations)
    task_count = len(tasks)
    avg_reward = (sum(rewards) / len(rewards)) if rewards else None
    pass_rate = (success_count / sim_count) if sim_count > 0 else None

    return {
        "path": str(path),
        "timestamp": meta["timestamp"],
        "mode": meta["mode"],
        "domain": domain,
        "agent_model": agent_info.get("llm"),
        "user_model": user_info.get("llm"),
        "num_trials": info.get("num_trials"),
        "max_steps": info.get("max_steps"),
        "max_errors": info.get("max_errors"),
        "tasks_declared": task_count,
        "simulations": sim_count,
        "success_count": success_count,
        "avg_reward": avg_reward,
        "pass_rate": pass_rate,
        "termination_summary": ", ".join(
            f"{k}:{v}" for k, v in sorted(termination_counter.items())
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate tau2 simulation JSON files into a CSV with one row per timestamp "
            "and one set of columns per domain."
        )
    )
    parser.add_argument(
        "--sim-dir",
        type=Path,
        default=DEFAULT_SIM_DIR,
        help="Directory containing simulation JSON files.",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="*.json",
        help="Glob pattern used to find simulation files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--sort",
        choices=["asc", "desc"],
        default="asc",
        help="Sort order by timestamp.",
    )
    args = parser.parse_args()

    sim_dir = args.sim_dir
    files = sorted(sim_dir.glob(args.glob))
    if not files:
        raise SystemExit(f"No files matched: {sim_dir / args.glob}")

    file_summaries = []
    skipped_files = 0
    for p in files:
        try:
            file_summaries.append(summarize_file(p))
        except Exception as e:
            skipped_files += 1
            print(f"Skipping file (not a valid simulation json): {p.name} ({e})")

    if not file_summaries:
        raise SystemExit(
            f"Matched {len(files)} json files, but none are valid simulation files."
        )

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in file_summaries:
        ts = s.get("timestamp")
        if ts is None:
            ts = datetime.fromtimestamp(Path(s["path"]).stat().st_mtime).strftime(
                "%Y%m%d_%H%M%S"
            )
        groups[ts].append(s)

    dynamic_domains = sorted({s["domain"] for s in file_summaries if s["domain"]})
    ordered_domains = []
    for d in TARGET_DOMAINS:
        if d in dynamic_domains:
            ordered_domains.append(d)
    for d in dynamic_domains:
        if d not in ordered_domains:
            ordered_domains.append(d)

    rows = []
    for ts, items in groups.items():
        ts_dt = datetime.strptime(ts, "%Y%m%d_%H%M%S")
        row: dict[str, Any] = {
            "timestamp": ts,
            "time_readable": ts_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "modes": "|".join(sorted({str(i["mode"]) for i in items if i["mode"]})),
            "agent_models": "|".join(
                sorted({str(i["agent_model"]) for i in items if i["agent_model"]})
            ),
            "user_models": "|".join(
                sorted({str(i["user_model"]) for i in items if i["user_model"]})
            ),
            "file_count": len(items),
        }

        by_domain = {i["domain"]: i for i in items}
        for domain in ordered_domains:
            item = by_domain.get(domain)
            prefix = f"{domain}_"
            if item is None:
                row[prefix + "present"] = 0
                row[prefix + "tasks"] = ""
                row[prefix + "simulations"] = ""
                row[prefix + "success_count"] = ""
                row[prefix + "avg_reward"] = ""
                row[prefix + "pass_rate"] = ""
                row[prefix + "termination"] = ""
                row[prefix + "file"] = ""
            else:
                row[prefix + "present"] = 1
                row[prefix + "tasks"] = item["tasks_declared"]
                row[prefix + "simulations"] = item["simulations"]
                row[prefix + "success_count"] = item["success_count"]
                row[prefix + "avg_reward"] = (
                    "" if item["avg_reward"] is None else f"{item['avg_reward']:.6f}"
                )
                row[prefix + "pass_rate"] = (
                    "" if item["pass_rate"] is None else f"{item['pass_rate']:.6f}"
                )
                row[prefix + "termination"] = item["termination_summary"]
                row[prefix + "file"] = Path(item["path"]).name

        rows.append((ts_dt, row))

    rows.sort(key=lambda x: x[0], reverse=(args.sort == "desc"))
    final_rows = [r for _, r in rows]

    fieldnames = list(final_rows[0].keys())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_rows)

    print(f"Matched {len(files)} files, analyzed {len(file_summaries)} files.")
    if skipped_files > 0:
        print(f"Skipped {skipped_files} files.")
    print(f"Wrote CSV: {args.output}")
    print(f"Rows: {len(final_rows)}")


if __name__ == "__main__":
    main()
