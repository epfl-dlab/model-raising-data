"""Charter eval CLI: eval-generators / eval-judges / rank-* / list-runs / failures.

Usage:
    uv run python -m pipeline.charter.eval eval-generators [--run-id NAME] [--stage generate|judge] [--mode reflection|preflection] [overrides...]
    uv run python -m pipeline.charter.eval eval-judges     [--run-id NAME] [overrides...]
    uv run python -m pipeline.charter.eval rank-generators <run_id> [--json]
    uv run python -m pipeline.charter.eval rank-judges     <run_id> [--json]
    uv run python -m pipeline.charter.eval list-runs
    uv run python -m pipeline.charter.eval failures        <run_id> [--category api|parse]
    uv run python -m pipeline.charter.eval report          <run_id> [run_id2 ...] [--out PATH] [--source auto|generations|judgments]
    uv run python -m pipeline.charter.eval deploy-dashboard <user>/<space-name>
    uv run python -m pipeline.charter.eval retrieve-feedback <user>/<dataset> [--out PATH]
    uv run python -m pipeline.charter.eval normative-sample [--run-id NAME] [--n-items 100] [--out PATH]

OmegaConf-style dotlist overrides work the same as in charter.improve:
    uv run python -m pipeline.charter.eval eval-generators charter.eval.generator_eval.n_items=20

The --stage flag for eval-generators lets you run generation and judging separately:
    uv run python -m pipeline.charter.eval eval-generators --run-id my-run --stage generate
    uv run python -m pipeline.charter.eval eval-generators --run-id my-run --stage judge

The --mode flag restricts to a single pipeline (reflection or preflection):
    uv run python -m pipeline.charter.eval eval-generators --run-id my-run --mode reflection
"""

from __future__ import annotations

import datetime
import json
import sys

from pipeline.config import load_config
from pipeline.log import logger
from pipeline.charter.eval.eval_generators import _eval_root


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")


def _split_run_id_and_overrides(
    args: list[str],
) -> tuple[str | None, str | None, str | None, list[str]]:
    """Pull --run-id NAME, --stage NAME, and --mode NAME out of args.

    Returns ``(run_id, stage, mode, remaining_overrides)``.
    """
    run_id: str | None = None
    stage: str | None = None
    mode: str | None = None
    rest: list[str] = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--run-id" and i + 1 < len(args):
            run_id = args[i + 1]
            i += 2
            continue
        if a == "--stage" and i + 1 < len(args):
            stage = args[i + 1]
            i += 2
            continue
        if a == "--mode" and i + 1 < len(args):
            mode = args[i + 1]
            i += 2
            continue
        rest.append(a)
        i += 1
    return run_id, stage, mode, rest


def cmd_eval_generators(args: list[str]) -> int:
    run_id, stage, mode, overrides = _split_run_id_and_overrides(args)
    if stage and stage not in ("generate", "judge"):
        print(f"Unknown --stage {stage!r}. Must be 'generate' or 'judge'.")
        return 2
    if mode and mode not in ("reflection", "preflection"):
        print(f"Unknown --mode {mode!r}. Must be 'reflection' or 'preflection'.")
        return 2
    cfg = load_config(overrides=overrides if overrides else None)
    if mode:
        cfg.charter.eval.generator_eval.mode = mode
    if not run_id:
        run_id = f"gen_eval_{_now_iso()}"
    from pipeline.charter.eval.eval_generators import run_generator_eval

    logger.info(
        "charter.eval eval-generators run_id={} stage={} mode={}",
        run_id,
        stage or "all",
        cfg.charter.eval.generator_eval.mode or "both",
    )
    run_generator_eval(cfg, run_id, stage=stage)
    print(f"\nDone. run_id={run_id}")
    return 0


def cmd_eval_judges(args: list[str]) -> int:
    run_id, _stage, _mode, overrides = _split_run_id_and_overrides(args)
    cfg = load_config(overrides=overrides if overrides else None)
    if not run_id:
        run_id = f"judge_eval_{_now_iso()}"
    from pipeline.charter.eval.eval_judges import run_judge_eval

    logger.info("charter.eval eval-judges run_id={}", run_id)
    run_judge_eval(cfg, run_id)
    print(f"\nDone. run_id={run_id}")
    return 0


def cmd_rank_generators(args: list[str]) -> int:
    as_json = "--json" in args
    run_ids = [a for a in args if a != "--json"]
    if not run_ids:
        print("Usage: rank-generators <run_id> [run_id2 ...] [--json]")
        return 2
    from pipeline.charter.eval import rank as rank_mod

    all_rows: list[dict] = []
    for run_id in run_ids:
        all_rows.extend(rank_mod.rank_generators(run_id))
    if as_json:
        print(json.dumps(all_rows, indent=2))
        return 0
    if not all_rows:
        print(f"No generators in {', '.join(run_ids)}")
        return 0
    print(
        f"{'Generator':<40} {'n_ok':>6} {'mean':>6} {'accept':>8} "
        f"{'gen_api':>8} {'gen_parse':>10} {'jud_api':>8} {'jud_parse':>10}"
    )
    for r in all_rows:
        fr = r["failure_rates"]
        print(
            f"{r['generator']:<40} {r['n_succeeded']:>6} "
            f"{r['mean_aggregate']:>6.3f} {r['accept_rate']:>7.1%} "
            f"{fr['gen_api']:>7.1%} {fr['gen_parse']:>9.1%} "
            f"{fr['judge_api']:>7.1%} {fr['judge_parse']:>9.1%}"
        )
    return 0


def cmd_rank_judges(args: list[str]) -> int:
    if not args:
        print("Usage: rank-judges <run_id> [--json]")
        return 2
    run_id = args[0]
    as_json = "--json" in args[1:]
    from pipeline.charter.eval import rank as rank_mod

    blocks = rank_mod.rank_judges(run_id)
    if as_json:
        print(json.dumps(blocks, indent=2))
        return 0
    for label, key in (("vs gold", "vs_gold"), ("vs human", "vs_human")):
        rows = blocks.get(key) or []
        print(f"\n=== judges {label} ===")
        if not rows:
            print("(empty)")
            continue
        print(
            f"{'Judge':<50} {'n_ok':>6} {'spearman':>9} {'pearson':>8} "
            f"{'conc':>6} {'kappa':>6} {'api%':>6} {'parse%':>7}"
        )
        for r in rows:
            fr = r.get("failure_rates", {})
            print(
                f"{r['judge']:<50} {r.get('n_succeeded', 0):>6} "
                f"{_fmt(r.get('spearman')):>9} {_fmt(r.get('pearson')):>8} "
                f"{_fmt(r.get('concordance')):>6} {_fmt(r.get('kappa')):>6} "
                f"{fr.get('api', 0.0):>5.1%} {fr.get('parse', 0.0):>6.1%}"
            )
    return 0


def cmd_report(args: list[str]) -> int:
    out = None
    source = "generations"
    run_ids: list[str] = []
    i = 0
    while i < len(args):
        if args[i] == "--out" and i + 1 < len(args):
            out = args[i + 1]
            i += 2
            continue
        if args[i] == "--source" and i + 1 < len(args):
            source = args[i + 1]
            i += 2
            continue
        run_ids.append(args[i])
        i += 1
    if source not in ("auto", "generations", "judgments"):
        print("Unknown --source. Must be auto, generations, or judgments.")
        return 2
    if not run_ids:
        print("Usage: report <run_id> [run_id2 ...] [--out PATH] [--source auto|generations|judgments]")
        return 2
    from pipeline.charter.eval.report import DEFAULT_CARDS_PATH, write_cards

    out_path = out or DEFAULT_CARDS_PATH
    n = write_cards(run_ids, out_path, source=source)
    print(f"Wrote {n} cards from {', '.join(run_ids)} -> {out_path}")
    return 0


def cmd_deploy_dashboard(args: list[str]) -> int:
    if not args:
        print("Usage: deploy-dashboard <user>/<space-name> [--folder DIR]")
        return 2
    space_id = args[0]
    folder = "dashboard"
    if "--folder" in args:
        folder = args[args.index("--folder") + 1]
    from pipeline.charter.eval.report import deploy_space

    deploy_space(space_id, folder)
    print(f"Deployed {folder} -> https://huggingface.co/spaces/{space_id}")
    return 0


def cmd_retrieve_feedback(args: list[str]) -> int:
    if not args:
        print("Usage: retrieve-feedback <user>/<dataset> [--out PATH]")
        return 2
    dataset = args[0]
    out = None
    if "--out" in args:
        out = args[args.index("--out") + 1]
    from pipeline.config import DATA_DIR
    from pipeline.charter.eval.report import retrieve_feedback, summarize_feedback

    local_dir = DATA_DIR / "pipeline" / "feedback" / dataset.replace("/", "__")
    rows = retrieve_feedback(dataset, local_dir)
    out_path = out or (local_dir.parent / f"{dataset.replace('/', '__')}_feedback_latest.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    s = summarize_feedback(rows)
    agree = f"{s['agreement']:.1%}" if s["agreement"] is not None else "-"
    print(f"{s['n']} verdicts ({s['accept']} accept / {s['reject']} reject) -> {out_path}")
    print(f"judge agreement (n={s['n_vs_judge']}): {agree}")
    return 0


def cmd_normative_sample(args: list[str]) -> int:
    run_id = None
    n_items = 100
    out = None
    i = 0
    while i < len(args):
        if args[i] == "--run-id" and i + 1 < len(args):
            run_id = args[i + 1]
            i += 2
            continue
        if args[i] == "--n-items" and i + 1 < len(args):
            n_items = int(args[i + 1])
            i += 2
            continue
        if args[i] == "--out" and i + 1 < len(args):
            out = args[i + 1]
            i += 2
            continue
        print("Usage: normative-sample [--run-id NAME] [--n-items 100] [--out PATH]")
        return 2
    if run_id is None:
        run_id = f"normative_hierarchy_{_now_iso()}"

    from pipeline.config import CandidateModel
    from pipeline.charter.eval.eval_generators import run_generator_eval
    from pipeline.charter.eval.report import DEFAULT_CARDS_PATH, write_cards

    cfg = load_config()
    cfg.charter_path = "resources/NormativeHierarchyConstitution_v0.1.md"
    cfg.writing_guidelines_path = "resources/NormativeHierarchyAnnotationGuidelines_v0.1.md"
    cfg.charter.eval.generator_eval.n_items = n_items
    cfg.charter.eval.generator_eval.mode = "reflection"
    cfg.charter.eval.generator_eval.safety_values = [0, 1, 2, 3, 4]
    cfg.charter.eval.generator_eval.candidates = [
        CandidateModel(
            alias="qwen3.6-35b-a3b",
            api_name="qwen/qwen3.6-35b-a3b",
            hf_slug="Qwen/Qwen3.6-35B-A3B-FP8",
            endpoint="https://openrouter.ai/api/v1",
            prompt_reflection="generator_reflection_normative_hierarchy_v1.md",
            context_window_tokens=32768,
            include_reflection_3p=False,
        )
    ]
    run_generator_eval(cfg, run_id, stage="generate")
    out_path = out or DEFAULT_CARDS_PATH
    n = write_cards(
        [run_id],
        out_path,
        eval_dir=cfg.charter.eval.eval_dir,
        source="generations",
        charter_path=cfg.charter_path,
    )
    print(f"\nDone. run_id={run_id}")
    print(f"Wrote {n} dashboard cards -> {out_path}")
    return 0


def cmd_list_runs(args: list[str]) -> int:
    cfg = load_config()
    root = _eval_root(cfg)
    if not root.exists():
        print(f"No charter.eval root at {root}")
        return 0
    rows: list[dict] = []
    for run_dir in sorted(root.iterdir()):
        meta_path = run_dir / "metadata.json"
        if not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        rows.append(
            {
                "run_id": run_dir.name,
                "type": meta.get("type", "?"),
                "status": meta.get("status", "?"),
                "n_items": meta.get("n_items", "?"),
                "started_at": (meta.get("started_at") or "")[:19],
                "finished_at": (meta.get("finished_at") or "")[:19] or "—",
                "n_candidates": len(meta.get("candidates", [])),
            }
        )
    if not rows:
        print("No charter.eval runs.")
        return 0
    print(
        f"{'Run':<40} {'Type':<16} {'Status':<10} {'Items':>6} "
        f"{'Cands':>6} {'Started':<19}  {'Finished':<19}"
    )
    for r in rows:
        print(
            f"{r['run_id']:<40} {r['type']:<16} {r['status']:<10} "
            f"{str(r['n_items']):>6} {r['n_candidates']:>6} "
            f"{r['started_at']:<19}  {r['finished_at']:<19}"
        )
    return 0


def cmd_failures(args: list[str]) -> int:
    if not args:
        print("Usage: failures <run_id> [--category api|parse] [--stage NAME]")
        return 2
    run_id = args[0]
    category = None
    stage = None
    i = 1
    while i < len(args):
        if args[i] == "--category" and i + 1 < len(args):
            category = args[i + 1]
            i += 2
            continue
        if args[i] == "--stage" and i + 1 < len(args):
            stage = args[i + 1]
            i += 2
            continue
        i += 1

    cfg = load_config()
    root = _eval_root(cfg)
    failures_dir = root / run_id / "failures"
    if not failures_dir.exists():
        print(f"No failures dir for run {run_id}")
        return 0
    n_total = 0
    for path in sorted(failures_dir.glob("*.jsonl")):
        rows = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        if category:
            rows = [r for r in rows if r.get("category") == category]
        if stage:
            rows = [r for r in rows if r.get("stage") == stage]
        if not rows:
            continue
        print(f"\n=== {path.name} ({len(rows)} rows) ===")
        for r in rows[:50]:
            print(
                f"  {r.get('item_id', '?')} cat={r.get('category', '?')} "
                f"stage={r.get('stage', '?')} reason={r.get('reason', '?')} "
                f"attempt={r.get('attempt', '?')}"
            )
            raw = r.get("raw") or ""
            if raw:
                preview = raw[:200].replace("\n", " ")
                print(f"      raw: {preview}{'…' if len(raw) > 200 else ''}")
        if len(rows) > 50:
            print(f"  … {len(rows) - 50} more")
        n_total += len(rows)
    print(f"\nTotal failures shown: {n_total}")
    return 0


def _fmt(v) -> str:
    if v is None:
        return "—"
    try:
        return f"{v:.3f}"
    except (TypeError, ValueError):
        return str(v)


_DISPATCH = {
    "eval-generators": cmd_eval_generators,
    "eval-judges": cmd_eval_judges,
    "rank-generators": cmd_rank_generators,
    "rank-judges": cmd_rank_judges,
    "report": cmd_report,
    "deploy-dashboard": cmd_deploy_dashboard,
    "retrieve-feedback": cmd_retrieve_feedback,
    "normative-sample": cmd_normative_sample,
    "list-runs": cmd_list_runs,
    "failures": cmd_failures,
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(__doc__)
        return 2
    cmd = argv[0]
    handler = _DISPATCH.get(cmd)
    if handler is None:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        return 2
    return handler(argv[1:])


if __name__ == "__main__":
    sys.exit(main())
