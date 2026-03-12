"""Phase 2 dashboard pages: /pipeline and /pipeline/review routes."""

from __future__ import annotations

import statistics
import threading
from pathlib import Path

from nicegui import app, ui

from pipeline.config import CHARTER_PATH, AppConfig, load_config
from pipeline.dashboard import render_header
from pipeline.dashboard.shared import render_source_text
from pipeline.phase2.storage import (
    load_items_for_iteration,
    load_latest_reviews,
    load_reviews,
    load_runs,
    save_review,
)


def _load_charter() -> str:
    return CHARTER_PATH.read_text(encoding="utf-8")


def _compute_calibration(reviews: list[dict], items_by_key: dict) -> dict:
    """Compute judge-vs-human calibration metrics.

    Returns per-dimension correlations, aggregate correlation, and decision agreement.
    """
    paired_scores: dict[str, list[tuple[float, float]]] = {}
    aggregate_pairs: list[tuple[float, float]] = []
    decision_pairs: list[tuple[str, str]] = []

    for review in reviews:
        key = (review["item_id"], review["iteration"])
        item = items_by_key.get(key)
        if not item or not item.get("judgment"):
            continue
        judgment = item["judgment"]

        # Collect per-dimension pairs from both preflection and reflection sub-judgments
        for part in ("preflection", "reflection"):
            part_j = judgment.get(part, {})
            part_scores = part_j.get("scores", {})
            for dim, human_score in review["scores"].items():
                if dim in part_scores:
                    paired_scores.setdefault(f"{part}_{dim}", []).append(
                        (part_scores[dim], human_score)
                    )

        aggregate_pairs.append((judgment["aggregate"], review["aggregate"]))
        decision_pairs.append((judgment["decision"], review["decision"]))

    def _pearson(pairs: list[tuple[float, float]]) -> float | None:
        if len(pairs) < 3:
            return None
        x = [p[0] for p in pairs]
        y = [p[1] for p in pairs]
        mx = statistics.mean(x)
        my = statistics.mean(y)
        cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
        sx = (sum((xi - mx) ** 2 for xi in x)) ** 0.5
        sy = (sum((yi - my) ** 2 for yi in y)) ** 0.5
        if sx == 0 or sy == 0:
            return None
        return cov / (sx * sy)

    dim_corr = {dim: _pearson(pairs) for dim, pairs in paired_scores.items()}
    agg_corr = _pearson(aggregate_pairs)
    agreement = (
        sum(1 for j, h in decision_pairs if j == h) / len(decision_pairs)
        if decision_pairs
        else None
    )

    return {
        "dimension_correlations": dim_corr,
        "aggregate_correlation": agg_corr,
        "decision_agreement": agreement,
        "n_reviews": len(reviews),
        "n_paired": len(aggregate_pairs),
    }


@ui.page("/pipeline")
def pipeline_monitoring_page():
    """Pipeline monitoring dashboard: iteration table, trends, calibration."""
    viewer_id = app.storage.user.get("annotator_id", "")

    def pipeline_actions():
        ui.button("Review", icon="rate_review",
                  on_click=lambda: ui.navigate.to("/pipeline/review"),
                  ).classes("text-white").props("flat dense")

    render_header(viewer_id, active_phase=2, right_slot=pipeline_actions)

    runs = load_runs()
    all_reviews = load_reviews()

    # Build items index for calibration
    items_by_key: dict[tuple[str, int], dict] = {}
    for run in runs:
        for item in load_items_for_iteration(run["iteration"]):
            items_by_key[(item["item_id"], item["iteration"])] = item

    # --- Precompute per-iteration stats (used by charts + table) ---
    iter_stats: list[dict] = []
    for run in runs:
        it = run["iteration"]
        items = load_items_for_iteration(it)
        judged = [i for i in items if i.get("judgment")]
        n_acc = sum(1 for i in judged if i["judgment"]["decision"] == "accept")
        scores = [i["judgment"]["aggregate"] for i in judged]
        mean_s = statistics.mean(scores) if scores else 0
        accept_rate = round(n_acc / len(judged) * 100, 1) if judged else 0

        iter_reviews = [r for r in all_reviews if r["iteration"] == it]
        iter_items = {(i["item_id"], i["iteration"]): i for i in items}
        cal_iter = _compute_calibration(iter_reviews, iter_items)

        iter_stats.append({
            **run,
            "n_acc": n_acc,
            "n_rej": len(judged) - n_acc,
            "mean_score": mean_s,
            "accept_rate": accept_rate,
            "calibration": cal_iter,
        })

    # --- Judge Calibration Panel ---
    with ui.card().classes("w-full q-mx-md q-mt-md q-pa-md"):
        ui.label("Judge Calibration").classes("text-h6 text-weight-bold")
        cal = _compute_calibration(all_reviews, items_by_key)
        if cal["n_paired"] == 0:
            ui.label("No human reviews yet — submit reviews to see calibration metrics.").classes(
                "text-grey-6"
            )
        else:
            with ui.row().classes("gap-8"):
                with ui.column():
                    ui.label(f"Paired reviews: {cal['n_paired']}").classes("text-body2")
                    agg = cal["aggregate_correlation"]
                    ui.label(
                        f"Aggregate correlation: {agg:.3f}" if agg is not None else "Aggregate correlation: N/A"
                    ).classes("text-body2")
                    agr = cal["decision_agreement"]
                    ui.label(
                        f"Decision agreement: {agr:.1%}" if agr is not None else "Decision agreement: N/A"
                    ).classes("text-body2")
                with ui.column():
                    ui.label("Per-dimension correlations:").classes("text-body2 text-weight-bold")
                    for dim, corr in cal["dimension_correlations"].items():
                        val = f"{corr:.3f}" if corr is not None else "N/A"
                        ui.label(f"  {dim}: {val}").classes("text-body2")

    # --- Trend Charts ---
    if len(iter_stats) >= 2:
        iter_labels = [f"Iter {s['iteration']}" for s in iter_stats]
        accept_rates = [s["accept_rate"] for s in iter_stats]
        mean_scores = [round(s["mean_score"], 2) for s in iter_stats]
        calibration_corrs = [s["calibration"]["aggregate_correlation"] for s in iter_stats]

        with ui.row().classes("w-full q-mx-md q-mt-md gap-4"):
            with ui.card().classes("flex-1 q-pa-md"):
                ui.label("Acceptance Rate & Mean Score").classes("text-subtitle2 text-weight-bold")
                ui.echart({
                    "xAxis": {"type": "category", "data": iter_labels},
                    "yAxis": [
                        {"type": "value", "name": "Accept %", "min": 0, "max": 100, "position": "left"},
                        {"type": "value", "name": "Mean Score", "min": 1, "max": 5, "position": "right"},
                    ],
                    "series": [
                        {"name": "Accept %", "type": "line", "data": accept_rates, "yAxisIndex": 0,
                         "itemStyle": {"color": "#4caf50"}},
                        {"name": "Mean Score", "type": "line", "data": mean_scores, "yAxisIndex": 1,
                         "itemStyle": {"color": "#2196f3"}},
                    ],
                    "tooltip": {"trigger": "axis"},
                    "legend": {"bottom": 0},
                }).classes("w-full").style("height: 250px;")

            with ui.card().classes("flex-1 q-pa-md"):
                ui.label("Judge–Human Correlation").classes("text-subtitle2 text-weight-bold")
                has_any = any(c is not None for c in calibration_corrs)
                if not has_any:
                    ui.label("No human reviews yet.").classes("text-grey-6")
                else:
                    corr_data = [round(c, 3) if c is not None else None for c in calibration_corrs]
                    ui.echart({
                        "xAxis": {"type": "category", "data": iter_labels},
                        "yAxis": {"type": "value", "name": "Pearson r", "min": -1, "max": 1},
                        "series": [
                            {"name": "Aggregate Correlation", "type": "line", "data": corr_data,
                             "itemStyle": {"color": "#ff9800"},
                             "connectNulls": True},
                        ],
                        "tooltip": {"trigger": "axis"},
                    }).classes("w-full").style("height: 250px;")

    # --- Iteration Table ---
    with ui.card().classes("w-full q-mx-md q-mt-md q-pa-md"):
        ui.label("Iterations").classes("text-h6 text-weight-bold")
        if not runs:
            ui.label("No iterations yet.").classes("text-grey-6")
        else:
            columns = [
                {"name": "iteration", "label": "#", "field": "iteration", "sortable": True},
                {"name": "model", "label": "Model", "field": "model"},
                {"name": "gen_prompt", "label": "Gen Prompt", "field": "gen_prompt"},
                {"name": "judge_prompt", "label": "Judge Prompt", "field": "judge_prompt"},
                {"name": "n_items", "label": "Items", "field": "n_items"},
                {"name": "n_gold", "label": "Gold", "field": "n_gold"},
                {"name": "timestamp", "label": "Time", "field": "timestamp"},
            ]
            rows = [
                {
                    **s,
                    # Backward compat: old records have "model", new have "generator_model"
                    "model": s.get("generator_model", s.get("model", "unknown")),
                    "timestamp": s["timestamp"][:19],
                    "accept_reject": f"{s['n_acc']}/{s['n_rej']}",
                    "mean_score": f"{s['mean_score']:.2f}",
                }
                for s in iter_stats
            ]

            extra_cols = [
                {"name": "accept_reject", "label": "Accept/Reject", "field": "accept_reject"},
                {"name": "mean_score", "label": "Mean Score", "field": "mean_score"},
            ]
            ui.table(columns=columns + extra_cols, rows=rows, row_key="iteration").classes("w-full")

            # Analysis expansion per iteration
            for run in runs:
                with ui.expansion(f"Iteration {run['iteration']} Analysis").classes("w-full"):
                    ui.markdown(run.get("analysis", "No analysis recorded.")).classes("text-body2")

    # --- Autonomous Loop ---
    with ui.card().classes("w-full q-mx-md q-mt-md q-pa-md"):
        ui.label("Autonomous Loop").classes("text-h6 text-weight-bold")
        ui.label(
            "Runs generate→judge→improve cycles. A Claude subprocess improves prompts between iterations."
        ).classes("text-caption text-grey-7")

        loop_progress = ui.linear_progress(value=0, show_value=False).classes("w-full q-mt-sm")
        loop_phase_label = ui.label("").classes("text-caption text-grey-6")
        loop_error_label = ui.label("").classes("text-caption text-red")

        loop_results_container = ui.column().classes("w-full gap-1 q-mt-sm")

        with ui.expansion("Improver Log", icon="terminal").classes("w-full q-mt-sm"):
            improver_log_display = ui.code("", language="text").classes("w-full").style(
                "max-height: 400px; overflow-y: auto; font-size: 0.8em;"
            )

        def _tail_improver_log(n_lines: int = 50) -> str:
            from pipeline.phase2.loop import IMPROVER_LOG_PATH
            if not IMPROVER_LOG_PATH.exists():
                return ""
            lines = IMPROVER_LOG_PATH.read_text().splitlines()
            return "\n".join(lines[-n_lines:])

        def _poll_loop_status():
            from pipeline.phase2.loop import read_status
            st = read_status()
            if st is None:
                return
            total = st.get("total_iterations", 1) or 1
            current = st.get("loop_iteration", 0)
            loop_progress.set_value(current / total)
            phase = st.get("phase", "")
            loop_phase_label.set_text(
                f"Iteration {current}/{total} — {phase}"
                if st.get("running") else f"Loop {phase}"
            )
            if st.get("error"):
                loop_error_label.set_text(f"Error: {st['error']}")
            else:
                loop_error_label.set_text("")

            if phase == "improving" or (not st.get("running") and phase in ("done", "error")):
                improver_log_display.set_content(_tail_improver_log())

            results = st.get("results", [])
            loop_results_container.clear()
            if results:
                with loop_results_container:
                    cols = [
                        {"name": "iter", "label": "#", "field": "loop_iteration"},
                        {"name": "accepted", "label": "Accepted", "field": "n_accepted"},
                        {"name": "rejected", "label": "Rejected", "field": "n_rejected"},
                        {"name": "score", "label": "Mean Score", "field": "mean_score"},
                    ]
                    ui.table(columns=cols, rows=results, row_key="loop_iteration").classes("w-full")

            if not st.get("running"):
                loop_timer.active = False
                loop_btn.enable()
                single_btn.enable()

        loop_timer = ui.timer(3.0, _poll_loop_status, active=False)

        def start_loop():
            loop_btn.disable()
            single_btn.disable()
            loop_error_label.set_text("")
            loop_phase_label.set_text("Starting loop...")
            loop_timer.active = True

            def _thread():
                import asyncio as _asyncio
                from pipeline.phase2.loop import run_loop
                cfg = load_config()
                _asyncio.run(run_loop(n_iterations=cfg.phase2.loop.n_iterations, cfg=cfg))

            threading.Thread(target=_thread, daemon=True).start()

        cfg_for_label = load_config()
        loop_btn = ui.button(
            f"Start Loop ({cfg_for_label.phase2.loop.n_iterations} iterations)",
            on_click=start_loop,
            color="primary",
        )

        # Check if a loop is already running
        from pipeline.phase2.loop import read_status as _read_initial
        _initial = _read_initial()
        if _initial and _initial.get("running"):
            loop_btn.disable()
            loop_timer.active = True
            loop_phase_label.set_text("Loop in progress...")

    # --- Single Iteration ---
    with ui.card().classes("w-full q-mx-md q-mt-md q-pa-md"):
        ui.label("Run Single Iteration").classes("text-h6 text-weight-bold")
        ui.label("Runs a single generate→judge iteration with current config.").classes(
            "text-caption text-grey-7"
        )

        def start_iteration():
            single_btn.disable()
            single_status.set_text("Running iteration...")

            def _run():
                import asyncio as _asyncio
                from pipeline.phase2.run import run_iteration

                cfg = load_config()
                result = _asyncio.run(run_iteration(cfg))
                return result

            def _done():
                single_btn.enable()
                single_status.set_text("Done! Refresh page to see results.")
                ui.notify("Iteration complete", type="positive")

            def _thread():
                try:
                    _run()
                    _done()
                except Exception as e:
                    single_status.set_text(f"Error: {e}")
                    single_btn.enable()

            threading.Thread(target=_thread, daemon=True).start()

        single_btn = ui.button("Start Iteration", on_click=start_iteration, color="secondary")
        single_status = ui.label("").classes("text-caption text-grey-6")


@ui.page("/pipeline/review")
def pipeline_review_page():
    """Human review of LLM-generated reflections with per-dimension scoring."""
    viewer_id = app.storage.user.get("annotator_id", "")
    if not viewer_id:
        ui.navigate.to("/")
        return

    def review_actions():
        ui.button("Dashboard", icon="dashboard",
                  on_click=lambda: ui.navigate.to("/pipeline"),
                  ).classes("text-white").props("flat dense")

    render_header(viewer_id, active_phase=2, right_slot=review_actions)

    runs = load_runs()
    if not runs:
        with ui.column().classes("absolute-center items-center"):
            ui.label("No iterations to review yet.").classes("text-h6 text-grey-6")
        return

    charter_text = _load_charter()
    cfg = load_config()
    dimensions = cfg.phase2.scoring.dimensions

    # State
    state = {"iteration": runs[-1]["iteration"], "pos": 0}

    # --- Iteration selector ---
    with ui.row().classes("q-px-md q-mt-md items-center gap-4"):
        iter_options = [r["iteration"] for r in runs]
        iter_select = ui.select(
            options=iter_options,
            value=state["iteration"],
            label="Iteration",
        ).classes("w-32")

        sort_select = ui.select(
            options=["Low score first", "High score first", "Default order"],
            value="Low score first",
            label="Sort",
        ).classes("w-48")

    # --- Main split panel ---
    with ui.splitter(value=35).classes("w-full").style("height: calc(100vh - 120px)") as splitter:
        # Left: source text + charter
        with splitter.before:
            with ui.column().classes("w-full p-4 gap-2").style(
                "position: sticky; top: 0; height: calc(100vh - 120px); overflow-y: auto;"
            ):
                ui.label("Source Text + Charter").classes("text-h6 text-weight-bold")
                source_html = ui.html("").style(
                    "max-height: 40%; overflow-y: auto; border: 1px solid #333; "
                    "border-radius: 4px; padding: 12px; line-height: 1.7; "
                    "font-family: Georgia, serif; white-space: pre-wrap; font-size: 0.95em;"
                )
                ui.separator()
                ui.label("Charter").classes("text-subtitle2 text-weight-bold")
                ui.markdown(charter_text, extras=["tables"]).classes("text-body2").style(
                    "flex: 1; overflow-y: auto; padding: 8px; line-height: 1.6;"
                )

        # Right: LLM generation + judge scores + review form
        with splitter.after:
            with ui.column().classes("w-full p-4 gap-2").style("overflow-y: auto;"):
                with ui.row().classes("items-center gap-4"):
                    nav_label = ui.label().classes("text-subtitle1 text-weight-medium")
                    subset_badge = ui.badge("").props("outline")
                    gold_badge = ui.badge("").props("outline color=orange")
                    ui.space()
                    ui.button(icon="arrow_back", on_click=lambda: navigate(-1)).props("flat dense")
                    ui.button(icon="arrow_forward", on_click=lambda: navigate(1)).props("flat dense")

                # LLM generation display
                gen_section = ui.column().classes("w-full gap-2")

                ui.separator()

                # Judge scores display
                judge_section = ui.column().classes("w-full gap-1")

                ui.separator()

                # Human annotation (for gold items)
                gold_section = ui.column().classes("w-full gap-1")

                ui.separator()

                # Review form
                ui.label("Your Review").classes("text-subtitle2 text-weight-bold")
                score_inputs: dict[str, ui.slider] = {}
                for dim in dimensions:
                    with ui.row().classes("items-center gap-2 w-full"):
                        ui.label(dim.replace("_", " ").title()).classes("w-40")
                        slider = ui.slider(min=1, max=5, value=3).classes("flex-1")
                        score_label = ui.label("3").classes("w-8")
                        slider.on("update:model-value", lambda e, lbl=score_label: lbl.set_text(str(int(e.args))))
                        score_inputs[dim] = slider

                decision_select = ui.select(
                    options=["accept", "reject"],
                    value="accept",
                    label="Decision",
                ).classes("w-48")

                notes_input = ui.textarea(
                    placeholder="Notes (optional)...",
                ).classes("w-full").props("outlined")

                with ui.row().classes("w-full justify-end"):
                    ui.button("Submit Review", on_click=lambda: submit_review(), color="primary")

    def get_sorted_items() -> list[dict]:
        items = load_items_for_iteration(state["iteration"])
        judged = [i for i in items if i.get("judgment")]
        sort = sort_select.value
        if sort == "Low score first":
            judged.sort(key=lambda i: i["judgment"]["aggregate"])
        elif sort == "High score first":
            judged.sort(key=lambda i: -i["judgment"]["aggregate"])
        return judged

    def current_items_list() -> list[dict]:
        return get_sorted_items()

    def update_display():
        items = current_items_list()
        if not items:
            nav_label.set_text("No judged items in this iteration")
            return

        state["pos"] = max(0, min(state["pos"], len(items) - 1))
        item = items[state["pos"]]

        nav_label.set_text(f"Item {state['pos'] + 1} / {len(items)}")
        subset_badge.set_text(item["subset"])
        gold_badge.set_text("GOLD" if item.get("is_gold") else "")
        gold_badge.set_visibility(item.get("is_gold", False))

        source_html.set_content(render_source_text(item["text"], item["reflection_point"]))

        gen_section.clear()
        with gen_section:
            ui.label("LLM Generation").classes("text-subtitle2 text-weight-bold")
            ui.label("Analysis").classes("text-overline text-grey-7")
            ui.label(item.get("analysis", "")).classes("text-body2").style("white-space: pre-wrap;")
            ui.label("Preflection").classes("text-overline text-grey-7")
            ui.label(item.get("preflection", "")).classes("text-body2").style("white-space: pre-wrap;")
            ui.label("Reflection").classes("text-overline text-grey-7")
            ui.label(item.get("reflection", "")).classes("text-body2").style("white-space: pre-wrap;")
            elements = item.get("charter_elements", [])
            if elements:
                ui.label("Charter Elements").classes("text-overline text-grey-7")
                with ui.row().classes("gap-1"):
                    for eid in elements:
                        ui.badge(eid, color="blue-grey-3").props("outline")

        judge_section.clear()
        with judge_section:
            judgment = item.get("judgment", {})
            if judgment:
                ui.label("Judge Scores").classes("text-subtitle2 text-weight-bold")
                with ui.row().classes("gap-4"):
                    ui.badge(
                        f"Aggregate: {judgment['aggregate']:.1f}",
                        color="green" if judgment["decision"] == "accept" else "red",
                    )
                    ui.badge(judgment["decision"].upper(), color="green" if judgment["decision"] == "accept" else "red")

                for part in ("preflection", "reflection"):
                    part_j = judgment.get(part, {})
                    if not part_j:
                        continue
                    ui.label(f"{part.title()} ({part_j.get('aggregate', 0):.1f})").classes(
                        "text-overline text-grey-7 q-mt-sm"
                    )
                    with ui.row().classes("gap-2"):
                        for dim, score in part_j.get("scores", {}).items():
                            color = "green" if score >= 4 else ("orange" if score >= 3 else "red")
                            ui.badge(f"{dim}: {score}", color=color)
                    ui.label(part_j.get("reasoning", "")).classes("text-body2").style(
                        "white-space: pre-wrap; font-size: 0.9em;"
                    )

        gold_section.clear()
        with gold_section:
            if item.get("is_gold"):
                _show_gold_annotation(item["item_id"])

        # Pre-fill from existing review
        latest_reviews = load_latest_reviews()
        existing = latest_reviews.get((item["item_id"], state["iteration"], viewer_id))
        if existing:
            for dim, slider in score_inputs.items():
                slider.set_value(existing["scores"].get(dim, 3))
            decision_select.set_value(existing["decision"])
            notes_input.set_value(existing.get("notes", ""))
        else:
            for slider in score_inputs.values():
                slider.set_value(3)
            decision_select.set_value("accept")
            notes_input.set_value("")

    def _show_gold_annotation(item_id: str):
        """Display the human annotation for a gold item."""
        from pipeline.phase1.storage import load_latest_annotations

        annotations = load_latest_annotations()
        gold_records = [v for (iid, _), v in annotations.items() if iid == item_id]
        if not gold_records:
            ui.label("No human annotations found for this gold item.").classes("text-grey-6")
            return

        ui.label("Human Annotation (Gold)").classes("text-subtitle2 text-weight-bold")
        for rec in gold_records:
            with ui.card().classes("w-full q-pa-sm"):
                ui.label(f"Annotator: {rec['annotator_id']}").classes("text-caption text-grey-6")
                ui.label("Analysis").classes("text-overline text-grey-7")
                ui.label(rec["analysis"]).classes("text-body2").style("white-space: pre-wrap;")
                ui.label("Preflection").classes("text-overline text-grey-7")
                ui.label(rec["preflection"]).classes("text-body2").style("white-space: pre-wrap;")
                ui.label("Reflection").classes("text-overline text-grey-7")
                ui.label(rec["reflection"]).classes("text-body2").style("white-space: pre-wrap;")

    def navigate(delta: int):
        items = current_items_list()
        new_pos = state["pos"] + delta
        if 0 <= new_pos < len(items):
            state["pos"] = new_pos
            update_display()

    def submit_review():
        items = current_items_list()
        if not items:
            return
        item = items[state["pos"]]
        scores = {dim: int(slider.value) for dim, slider in score_inputs.items()}
        aggregate = statistics.mean(scores.values())
        save_review(
            item_id=item["item_id"],
            iteration=state["iteration"],
            reviewer_id=viewer_id,
            scores=scores,
            aggregate=aggregate,
            decision=decision_select.value,
            notes=notes_input.value.strip(),
        )
        ui.notify("Review saved!", type="positive")
        navigate(1)

    def on_iteration_change(e):
        state["iteration"] = e.args
        state["pos"] = 0
        update_display()

    def on_sort_change(_):
        state["pos"] = 0
        update_display()

    iter_select.on("update:model-value", on_iteration_change)
    sort_select.on("update:model-value", on_sort_change)
    update_display()
