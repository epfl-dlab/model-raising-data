"""CLI for submitting and monitoring jobs on CSCS clusters via FirecREST.

Usage:
    uv run python slurm/cli.py submit --script path/to/script.sh --cluster clariden --account a141 --working-dir /users/jminder/...
    uv run python slurm/cli.py submit --script-str "#!/bin/bash -l\n..." --cluster clariden --account a141
    uv run python slurm/cli.py status --cluster clariden [--jobid 12345]
    uv run python slurm/cli.py cancel --cluster clariden --jobid 12345
    uv run python slurm/cli.py logs --cluster clariden --path /users/.../logs/12345
"""

import argparse
import sys
from datetime import datetime, timezone

from slurm.firecrest_client import FirecrestClient


def cmd_submit(client: FirecrestClient, args: argparse.Namespace):
    """Submit a job script to the cluster."""
    if args.script:
        with open(args.script) as f:
            script = f.read()
    elif args.script_str:
        script = args.script_str
    else:
        print("Error: either --script or --script-str is required", file=sys.stderr)
        sys.exit(1)

    # Ensure login shell for FirecREST compatibility
    if script.startswith("#!/bin/bash") and "-l" not in script.splitlines()[0]:
        script = script.replace("#!/bin/bash", "#!/bin/bash -l", 1)

    result = client.submit(
        cluster=args.cluster,
        script=script,
        working_dir=args.working_dir,
        account=args.account,
    )
    job_id = result.get("jobId", "unknown")
    print(f"Job submitted: {job_id}")
    return job_id


def _format_job(job: dict) -> str:
    """Format a single job dict for display."""
    status = job.get("status", {})
    state = status.get("state", "UNKNOWN")
    reason = status.get("stateReason", "")

    time_info = job.get("time", {})
    elapsed = time_info.get("elapsed", 0)
    limit = time_info.get("limit")
    start_ts = time_info.get("start")

    elapsed_str = f"{elapsed // 60}m{elapsed % 60}s" if elapsed else "-"
    limit_str = f"{limit // 60}m" if limit else "-"
    start_str = (
        datetime.fromtimestamp(start_ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        if start_ts
        else "-"
    )

    lines = [
        f"  Job ID:    {job.get('jobId', 'N/A')}",
        f"  Name:      {job.get('name', 'N/A')}",
        f"  State:     {state}" + (f" ({reason})" if reason and reason != "None" else ""),
        f"  Nodes:     {job.get('nodes', 'N/A')} ({job.get('allocationNodes', '?')} allocated)",
        f"  Partition: {job.get('partition', 'N/A')}",
        f"  Account:   {job.get('account', 'N/A')}",
        f"  Started:   {start_str}",
        f"  Elapsed:   {elapsed_str} / {limit_str}",
    ]
    return "\n".join(lines)


def cmd_status(client: FirecrestClient, args: argparse.Namespace):
    """Query job status."""
    jobs = client.job_info(cluster=args.cluster, jobid=args.jobid)
    if not jobs:
        print("No jobs found.")
        return

    for i, job in enumerate(jobs):
        if i > 0:
            print()
        print(_format_job(job))


def cmd_cancel(client: FirecrestClient, args: argparse.Namespace):
    """Cancel a job."""
    client.cancel(cluster=args.cluster, jobid=args.jobid)
    print(f"Cancel request sent for job {args.jobid}")


def cmd_logs(client: FirecrestClient, args: argparse.Namespace):
    """List and optionally read log files from a remote path."""
    files = client.list_files(cluster=args.cluster, path=args.path)
    if not files:
        print(f"No files found at {args.path}")
        return

    filenames = sorted(f["name"] for f in files)

    if args.read is not None:
        # Read specified files (or all if --read without value)
        targets = args.read if args.read else filenames
        for fname in targets:
            if fname not in filenames:
                print(f"--- {fname}: not found ---")
                continue
            full_path = f"{args.path.rstrip('/')}/{fname}"
            print(f"--- {fname} ---")
            content = client.head(cluster=args.cluster, path=full_path, num_lines=args.lines)
            print(content)
            print()
    else:
        print(f"Files in {args.path}:")
        for fname in filenames:
            print(f"  {fname}")


def main():
    parser = argparse.ArgumentParser(description="FirecREST CLI for CSCS clusters")
    parser.add_argument("--cluster", default="clariden", help="Cluster name (default: clariden)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- submit ---
    p_submit = subparsers.add_parser("submit", help="Submit a batch job")
    p_submit.add_argument("--script", help="Path to a local batch script file")
    p_submit.add_argument("--script-str", help="Inline script string")
    p_submit.add_argument("--working-dir", required=True, help="Remote working directory")
    p_submit.add_argument("--account", required=True, help="Slurm account")

    # --- status ---
    p_status = subparsers.add_parser("status", help="Query job status")
    p_status.add_argument("--jobid", default=None, help="Specific job ID (omit for all jobs)")

    # --- cancel ---
    p_cancel = subparsers.add_parser("cancel", help="Cancel a job")
    p_cancel.add_argument("--jobid", required=True, help="Job ID to cancel")

    # --- logs ---
    p_logs = subparsers.add_parser("logs", help="List/read remote log files")
    p_logs.add_argument("--path", required=True, help="Remote log directory path")
    p_logs.add_argument(
        "--read", nargs="*", default=None,
        help="Read log files (specify names, or omit for all)",
    )
    p_logs.add_argument("--lines", type=int, default=200, help="Number of lines to read (default: 200)")

    args = parser.parse_args()
    client = FirecrestClient()

    match args.command:
        case "submit":
            cmd_submit(client, args)
        case "status":
            cmd_status(client, args)
        case "cancel":
            cmd_cancel(client, args)
        case "logs":
            cmd_logs(client, args)


if __name__ == "__main__":
    main()
