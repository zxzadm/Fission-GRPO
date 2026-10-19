#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def run_command(cmd: list[str], timeout: int = 10) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except Exception as exc:
        return 1, "", str(exc)


def read_text(path: Path) -> str:
    try:
        return path.read_text()
    except Exception as exc:
        return f"<unavailable: {exc}>"


def read_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except Exception:
        return b""


def proc_exists(pid: int) -> bool:
    return Path(f"/proc/{pid}").exists()


def parse_proc_io(pid: int) -> dict[str, int]:
    result: dict[str, int] = {}
    text = read_text(Path(f"/proc/{pid}/io"))
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        try:
            result[key.strip()] = int(value)
        except ValueError:
            continue
    return result


def parse_proc_stat(pid: int) -> dict[str, int]:
    content = read_text(Path(f"/proc/{pid}/stat"))
    if content.startswith("<unavailable"):
        return {}
    right = content.rsplit(") ", 1)
    if len(right) != 2:
        return {}
    fields = right[1].split()
    if len(fields) < 15:
        return {}
    try:
        return {
            "state_code": fields[0],
            "minflt": int(fields[7]),
            "majflt": int(fields[9]),
            "utime": int(fields[11]),
            "stime": int(fields[12]),
            "num_threads": int(fields[17]),
        }
    except ValueError:
        return {}


def list_threads(pid: int) -> list[dict[str, str]]:
    task_root = Path(f"/proc/{pid}/task")
    rows = []
    for task in sorted(task_root.iterdir(), key=lambda p: int(p.name)):
        status = read_text(task / "status")
        name = ""
        state = ""
        for line in status.splitlines():
            if line.startswith("Name:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("State:"):
                state = line.split(":", 1)[1].strip()
        wchan = read_text(task / "wchan").strip()
        rows.append(
            {
                "tid": task.name,
                "name": name,
                "state": state,
                "wchan": wchan,
            }
        )
    return rows


def list_open_files(pid: int, limit: int = 30) -> list[str]:
    fd_root = Path(f"/proc/{pid}/fd")
    rows = []
    for fd in sorted(fd_root.iterdir(), key=lambda p: int(p.name)):
        try:
            target = os.readlink(fd)
        except OSError as exc:
            target = f"<unavailable: {exc}>"
        rows.append(f"{fd.name}: {target}")
        if len(rows) >= limit:
            break
    return rows


def find_pid(grep: str) -> int | None:
    code, out, err = run_command(
        ["ps", "-eo", "pid=,etimes=,pcpu=,pmem=,args="],
        timeout=10,
    )
    if code != 0:
        raise RuntimeError(err or "failed to run ps")
    matches = []
    needle = grep.lower()
    for line in out.splitlines():
        lower = line.lower()
        if needle not in lower:
            continue
        if "diagnose_bfcl_hang.py" in lower:
            continue
        parts = line.strip().split(None, 4)
        if len(parts) < 5:
            continue
        pid, etimes, pcpu, pmem, args = parts
        matches.append(
            {
                "pid": int(pid),
                "etimes": int(etimes),
                "pcpu": pcpu,
                "pmem": pmem,
                "args": args,
            }
        )
    if not matches:
        return None
    matches.sort(key=lambda item: item["etimes"], reverse=True)
    print("Matched processes:", file=sys.stderr)
    for item in matches[:10]:
        print(
            f"  pid={item['pid']} etimes={item['etimes']} pcpu={item['pcpu']} pmem={item['pmem']} args={item['args']}",
            file=sys.stderr,
        )
    return matches[0]["pid"]


def newest_files(root: Path, limit: int = 10) -> list[tuple[float, int, Path]]:
    rows = []
    if not root.exists():
        return rows
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        rows.append((stat.st_mtime, stat.st_size, path))
    rows.sort(reverse=True)
    return rows[:limit]


def maybe_dump_with_tool(pid: int, tool: str) -> str:
    if shutil.which(tool) is None:
        return f"{tool}: not installed"
    if tool == "py-spy":
        code, out, err = run_command(["py-spy", "dump", "--pid", str(pid)], timeout=20)
        return out or err or f"py-spy exit={code}"
    if tool == "pstack":
        code, out, err = run_command(["pstack", str(pid)], timeout=20)
        return out or err or f"pstack exit={code}"
    if tool == "gdb":
        code, out, err = run_command(
            ["gdb", "-p", str(pid), "-batch", "-ex", "thread apply all bt"],
            timeout=20,
        )
        return out or err or f"gdb exit={code}"
    if tool == "strace":
        code, out, err = run_command(
            ["strace", "-p", str(pid), "-f", "-tt", "-T", "-e", "trace=write,futex,openat,close,fsync", "-o", "/tmp/bfcl_strace.txt"],
            timeout=10,
        )
        note = out or err or f"strace exit={code}"
        trace = read_text(Path("/tmp/bfcl_strace.txt"))
        if trace and not trace.startswith("<unavailable"):
            return note + "\n" + trace[-8000:]
        return note
    return f"{tool}: unsupported"


def print_header(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose a hanging `bfcl generate` Python process on Linux."
    )
    parser.add_argument("--pid", type=int, default=None, help="Target process PID.")
    parser.add_argument(
        "--grep",
        default="bfcl generate",
        help="Find target PID by matching process args when --pid is omitted.",
    )
    parser.add_argument(
        "--result-dir",
        default=None,
        help="Optional BFCL result directory to inspect newest output files.",
    )
    parser.add_argument(
        "--tail-log",
        default=None,
        help="Optional log file to tail.",
    )
    parser.add_argument(
        "--sample-seconds",
        type=int,
        default=15,
        help="Duration to sample CPU/IO deltas.",
    )
    parser.add_argument(
        "--sample-interval",
        type=int,
        default=5,
        help="Sampling interval in seconds.",
    )
    parser.add_argument(
        "--try-py-spy",
        action="store_true",
        help="Try `py-spy dump --pid` if installed.",
    )
    parser.add_argument(
        "--try-pstack",
        action="store_true",
        help="Try `pstack` or fallback `gdb` if installed.",
    )
    parser.add_argument(
        "--try-strace",
        action="store_true",
        help="Try a short `strace -p` attach if installed and permitted.",
    )
    args = parser.parse_args()

    pid = args.pid
    if pid is None:
        pid = find_pid(args.grep)
        if pid is None:
            print(f"No process matched grep={args.grep!r}", file=sys.stderr)
            return 1

    if not proc_exists(pid):
        print(f"PID {pid} does not exist", file=sys.stderr)
        return 1

    proc_root = Path(f"/proc/{pid}")

    print_header("Process")
    cmdline = read_bytes(proc_root / "cmdline").replace(b"\x00", b" ").decode(
        "utf-8", errors="replace"
    )
    print(f"pid: {pid}")
    print(f"cmdline: {cmdline}")
    print(read_text(proc_root / "status"))

    print_header("Kernel Wait")
    print(f"wchan: {read_text(proc_root / 'wchan').strip()}")
    print(read_text(proc_root / "stack"))

    print_header("Threads")
    for row in list_threads(pid):
        print(
            f"tid={row['tid']} state={row['state']} wchan={row['wchan']} name={row['name']}"
        )

    print_header("Open Files")
    for row in list_open_files(pid):
        print(row)

    print_header("Sample")
    start_io = parse_proc_io(pid)
    start_stat = parse_proc_stat(pid)
    print(f"initial io: {start_io}")
    print(f"initial stat: {start_stat}")
    steps = max(1, args.sample_seconds // max(1, args.sample_interval))
    for idx in range(steps):
        time.sleep(args.sample_interval)
        if not proc_exists(pid):
            print(f"process exited during sampling at step {idx + 1}")
            break
        current_io = parse_proc_io(pid)
        current_stat = parse_proc_stat(pid)
        delta_write = current_io.get("write_bytes", 0) - start_io.get("write_bytes", 0)
        delta_read = current_io.get("read_bytes", 0) - start_io.get("read_bytes", 0)
        delta_utime = current_stat.get("utime", 0) - start_stat.get("utime", 0)
        delta_stime = current_stat.get("stime", 0) - start_stat.get("stime", 0)
        print(
            f"+{(idx + 1) * args.sample_interval:>3}s "
            f"read_bytes={delta_read} write_bytes={delta_write} "
            f"utime_ticks={delta_utime} stime_ticks={delta_stime} "
            f"threads={current_stat.get('num_threads', '?')} state={current_stat.get('state_code', '?')}"
        )

    if args.result_dir:
        print_header("Newest Result Files")
        for mtime, size, path in newest_files(Path(args.result_dir)):
            print(f"mtime={time.strftime('%F %T', time.localtime(mtime))} size={size} path={path}")

    if args.tail_log:
        print_header("Log Tail")
        code, out, err = run_command(["tail", "-n", "80", args.tail_log], timeout=10)
        print(out or err or f"tail exit={code}")

    if args.try_py_spy:
        print_header("py-spy")
        print(maybe_dump_with_tool(pid, "py-spy"))

    if args.try_pstack:
        print_header("Native Stack")
        if shutil.which("pstack"):
            print(maybe_dump_with_tool(pid, "pstack"))
        else:
            print(maybe_dump_with_tool(pid, "gdb"))

    if args.try_strace:
        print_header("strace")
        print(maybe_dump_with_tool(pid, "strace"))

    print_header("Diagnosis Hints")
    print("If write_bytes keeps increasing while CPU stays near zero, the process is likely draining the writer queue to result files.")
    print("If all threads sit in futex with no IO deltas, the process is likely blocked on a thread join or queue wait.")
    print("If one thread sits in openat/write/fsync repeatedly, it is likely stuck on slow filesystem IO.")
    print("If py-spy shows BaseHandler.write or writer_thread code, the hang is in result flushing, not model inference.")
    print("If py-spy shows wait(in_flight) with one worker still active, one test case is still hung despite tqdm reaching 100%.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
