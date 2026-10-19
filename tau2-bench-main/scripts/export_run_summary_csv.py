#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

TARGET_DOMAINS = ["retail", "airline", "telecom"]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SIM_DIR = PROJECT_ROOT / "data" / "simulations"
DEFAULT_OUTPUT_DIR = DEFAULT_SIM_DIR / "summaries"


def sanitize_tag(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return sanitized[:64] if sanitized else "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export one CSV summary for a single tau2 run tag, aggregating all "
            "selected domains into one wide row."
        )
    )
    parser.add_argument("--run-tag", required=True, help="Run tag, e.g. 20260312_144225")
    parser.add_argument(
        "--sim-dir",
        type=Path,
        default=DEFAULT_SIM_DIR,
        help="Directory containing simulation JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write the per-run summary CSV into.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Explicit output CSV path. Overrides --output-dir naming.",
    )
    parser.add_argument(
        "--mode",
        default=None,
        help="Optional mode filter, e.g. smoke or full.",
    )
    parser.add_argument(
        "--agent-label",
        default=None,
        help="Short label used in the output CSV filename.",
    )
    parser.add_argument(
        "--user-label",
        default=None,
        help="Short label used in the output CSV filename.",
    )
    return parser.parse_args()


def discover_run_files(sim_dir: Path, run_tag: str, mode: str | None) -> list[Path]:
    files = sorted(sim_dir.glob(f"qs_{run_tag}_*.json"))
    if mode is None:
        return files
    filtered = []
    needle = f"_{mode}"
    for path in files:
        if needle in path.stem:
            filtered.append(path)
    return filtered


def select_latest_per_domain(paths: list[Path]) -> tuple[dict[str, Path], dict[str, int]]:
    by_domain: dict[str, list[Path]] = {}
    for path in paths:
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        domain = (
            data.get("info", {})
            .get("environment_info", {})
            .get("domain_name")
        )
        if not domain:
            continue
        by_domain.setdefault(domain, []).append(path)

    selected: dict[str, Path] = {}
    duplicate_counts: dict[str, int] = {}
    for domain, domain_paths in by_domain.items():
        ordered = sorted(domain_paths, key=lambda p: p.stat().st_mtime, reverse=True)
        selected[domain] = ordered[0]
        duplicate_counts[domain] = len(ordered)
    return selected, duplicate_counts


def is_successful(reward: float | None) -> bool:
    if reward is None:
        return False
    return (1 - 1e-6) <= reward <= (1 + 1e-6)


def pass_hat_k(num_trials: int, success_count: int, k: int) -> float:
    if num_trials < k:
        raise ValueError(f"Number of trials {num_trials} is less than k {k}.")
    return math.comb(success_count, k) / math.comb(num_trials, k)


def summarize_results(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    info = data.get("info", {})
    simulations = data.get("simulations", []) or []
    tasks = data.get("tasks", []) or []
    termination_counter = Counter(
        str(sim.get("termination_reason", "unknown")) for sim in simulations
    )
    rewards = []
    agent_costs = []
    user_costs = []
    by_task: dict[str, list[bool]] = {}
    success_count = 0
    for sim in simulations:
        reward = sim.get("reward_info", {}).get("reward")
        if isinstance(reward, (int, float)):
            rewards.append(float(reward))
        success = is_successful(reward if isinstance(reward, (int, float)) else None)
        success_count += int(success)
        by_task.setdefault(str(sim.get("task_id")), []).append(success)

        agent_cost = sim.get("agent_cost")
        user_cost = sim.get("user_cost")
        if isinstance(agent_cost, (int, float)):
            agent_costs.append(float(agent_cost))
        if isinstance(user_cost, (int, float)):
            user_costs.append(float(user_cost))

    sim_count = len(simulations)
    success_rate = (success_count / sim_count) if sim_count > 0 else None
    avg_reward = (sum(rewards) / len(rewards)) if rewards else 0.0
    avg_agent_cost = (sum(agent_costs) / len(agent_costs)) if agent_costs else 0.0
    avg_user_cost = (sum(user_costs) / len(user_costs)) if user_costs else 0.0

    max_k = int(info.get("num_trials") or 0)
    if by_task:
        max_observed = min(len(v) for v in by_task.values())
        if max_k <= 0 or max_observed < max_k:
            max_k = max_observed
    pass_hat_ks: dict[int, float] = {}
    if max_k > 0 and by_task:
        for k in range(1, max_k + 1):
            values = []
            for successes in by_task.values():
                success_trials = sum(1 for ok in successes if ok)
                values.append(pass_hat_k(len(successes), success_trials, k))
            pass_hat_ks[k] = sum(values) / len(values)

    return {
        "path": path,
        "data": data,
        "sim_count": sim_count,
        "task_count": len(tasks),
        "success_count": success_count,
        "success_rate": success_rate,
        "avg_reward": avg_reward,
        "avg_agent_cost": avg_agent_cost,
        "avg_user_cost": avg_user_cost,
        "pass_hat_ks": pass_hat_ks,
        "termination_summary": ", ".join(
            f"{k}:{v}" for k, v in sorted(termination_counter.items())
        ),
    }


def build_output_path(
    output: Path | None,
    output_dir: Path,
    run_tag: str,
    mode: str | None,
    agent_label: str | None,
    user_label: str | None,
    selected_summaries: list[dict[str, Any]],
) -> Path:
    if output is not None:
        return output

    derived_mode = mode
    if derived_mode is None:
        derived_mode = "run"

    if agent_label is None:
        agent_models = sorted(
            {
                str(
                    summary["data"].get("info", {}).get("agent_info", {}).get("llm")
                    or "unknown"
                )
                for summary in selected_summaries
            }
        )
        agent_label = sanitize_tag("_".join(agent_models))
    else:
        agent_label = sanitize_tag(agent_label)

    if user_label is None:
        user_models = sorted(
            {
                str(
                    summary["data"].get("info", {}).get("user_info", {}).get("llm")
                    or "unknown"
                )
                for summary in selected_summaries
            }
        )
        user_label = sanitize_tag("_".join(user_models))
    else:
        user_label = sanitize_tag(user_label)

    stem = f"qs_{run_tag}_{derived_mode}_{agent_label}_user-{user_label}"
    return output_dir / f"{stem}.csv"


def maybe_format_float(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return f"{float(value):.6f}"
    return str(value)


def main() -> None:
    args = parse_args()
    files = discover_run_files(args.sim_dir, args.run_tag, args.mode)
    if not files:
        raise SystemExit(
            f"No simulation json files matched run_tag={args.run_tag} in {args.sim_dir}"
        )

    selected_by_domain, duplicate_counts = select_latest_per_domain(files)
    if not selected_by_domain:
        raise SystemExit(f"No valid simulation files found for run_tag={args.run_tag}")

    ordered_domains = list(TARGET_DOMAINS)
    for domain in sorted(selected_by_domain):
        if domain not in ordered_domains:
            ordered_domains.append(domain)

    summaries = {domain: summarize_results(path) for domain, path in selected_by_domain.items()}
    selected_summaries = [summaries[domain] for domain in ordered_domains if domain in summaries]

    pass_ks = sorted(
        {
            k
            for summary in selected_summaries
            for k in summary["pass_hat_ks"].keys()
        }
    )

    sample_info = selected_summaries[0]["data"].get("info", {})
    row: dict[str, Any] = {
        "run_tag": args.run_tag,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "mode": args.mode or "",
        "domain_count": len(selected_summaries),
        "domains_present": "|".join(
            [domain for domain in ordered_domains if domain in summaries]
        ),
        "agent_model": sample_info.get("agent_info", {}).get("llm", ""),
        "user_model": sample_info.get("user_info", {}).get("llm", ""),
    }

    for domain in ordered_domains:
        prefix = f"{domain}_"
        summary = summaries.get(domain)
        if summary is None:
            row[prefix + "present"] = 0
            row[prefix + "file"] = ""
            row[prefix + "duplicate_files"] = 0
            row[prefix + "tasks"] = ""
            row[prefix + "simulations"] = ""
            row[prefix + "num_trials"] = ""
            row[prefix + "max_steps"] = ""
            row[prefix + "max_errors"] = ""
            row[prefix + "success_count"] = ""
            row[prefix + "success_rate"] = ""
            row[prefix + "avg_reward"] = ""
            row[prefix + "avg_agent_cost"] = ""
            row[prefix + "avg_user_cost"] = ""
            row[prefix + "termination_summary"] = ""
            for k in pass_ks:
                row[f"{prefix}pass_hat_{k}"] = ""
            continue
        info = summary["data"].get("info", {})
        row[prefix + "present"] = 1
        row[prefix + "file"] = summary["path"].name
        row[prefix + "duplicate_files"] = duplicate_counts.get(domain, 1)
        row[prefix + "tasks"] = summary["task_count"]
        row[prefix + "simulations"] = summary["sim_count"]
        row[prefix + "num_trials"] = info.get("num_trials", "")
        row[prefix + "max_steps"] = info.get("max_steps", "")
        row[prefix + "max_errors"] = info.get("max_errors", "")
        row[prefix + "success_count"] = summary["success_count"]
        row[prefix + "success_rate"] = maybe_format_float(summary["success_rate"])
        row[prefix + "avg_reward"] = maybe_format_float(summary["avg_reward"])
        row[prefix + "avg_agent_cost"] = maybe_format_float(summary["avg_agent_cost"])
        row[prefix + "avg_user_cost"] = maybe_format_float(summary["avg_user_cost"])
        row[prefix + "termination_summary"] = summary["termination_summary"]
        for k in pass_ks:
            row[f"{prefix}pass_hat_{k}"] = maybe_format_float(
                summary["pass_hat_ks"].get(k)
            )

    output_path = build_output_path(
        output=args.output,
        output_dir=args.output_dir,
        run_tag=args.run_tag,
        mode=args.mode,
        agent_label=args.agent_label,
        user_label=args.user_label,
        selected_summaries=selected_summaries,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    print(f"Matched {len(files)} files, selected {len(selected_summaries)} domains.")
    print(f"Wrote run summary CSV: {output_path}")


if __name__ == "__main__":
    main()
