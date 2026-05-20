"""Tests for the SafeLM rephrasing baseline (pipeline.rephrasing_safelm + scale run)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from pipeline.charter.scale.runs import (
    RUNS,
    RUN_ALIASES,
    RunDefinition,
    _rephrasing_safelm_build_calls,
    _rephrasing_safelm_post_process,
    get_run,
)
from pipeline.rephrasing_safelm.templates import parse_templates, select_template


# Composite-prompt parsing ----------------------------------------------------


def _sample_composite(templates: list[tuple[str, str]]) -> str:
    """Helper: format ``[(name, body), ...]`` into a composite prompt file."""
    parts = ["# preamble (ignored)\n"]
    for name, body in templates:
        parts.append(f"\n<!-- TEMPLATE: {name} -->\n\n{body}\n")
    return "".join(parts)


class TestParseTemplates:
    def test_splits_all_seven_templates_from_v1(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "pipeline"
            / "rephrasing_safelm"
            / "prompts"
            / "rephrasing_safelm_v1.md"
        )
        templates = parse_templates(path.read_text(encoding="utf-8"))
        assert set(templates.keys()) == {
            "podcast",
            "textbook",
            "teacher_script",
            "ted_talk",
            "parent_child",
            "friends",
            "youtube_kids",
        }
        # Every template should be non-trivial (the SafeLM prompts run hundreds of chars).
        for name, body in templates.items():
            assert len(body) > 100, f"Template {name!r} suspiciously short ({len(body)} chars)"

    def test_strips_whitespace_around_name(self):
        text = "\n<!-- TEMPLATE:   podcast   -->\nbody text\n"
        templates = parse_templates(text)
        assert list(templates.keys()) == ["podcast"]

    def test_rejects_zero_templates(self):
        with pytest.raises(AssertionError, match="No <!-- TEMPLATE"):
            parse_templates("just some prose, no markers here")

    def test_rejects_duplicate_names(self):
        text = _sample_composite([("dup", "first"), ("dup", "second")])
        with pytest.raises(AssertionError, match="Duplicate template name"):
            parse_templates(text)

    def test_rejects_empty_body(self):
        text = "<!-- TEMPLATE: hollow -->\n\n<!-- TEMPLATE: filled -->\n\nactual content\n"
        with pytest.raises(AssertionError, match="empty body"):
            parse_templates(text)


# Per-doc template selection --------------------------------------------------


class TestSelectTemplate:
    TEMPLATES = {f"t{i}": f"body{i}" for i in range(7)}

    def test_deterministic_for_same_inputs(self):
        a = select_template(self.TEMPLATES, "doc_abc", seed=42)
        b = select_template(self.TEMPLATES, "doc_abc", seed=42)
        assert a == b

    def test_different_docs_can_get_different_templates(self):
        # With 7 templates and 100 docs, virtually certain to see >=2 distinct.
        names = {
            select_template(self.TEMPLATES, f"doc_{i}", seed=42)[0]
            for i in range(100)
        }
        assert len(names) > 1

    def test_returns_body_matching_name(self):
        name, body = select_template(self.TEMPLATES, "doc_xyz", seed=42)
        assert self.TEMPLATES[name] == body

    def test_covers_all_seven_templates_over_many_docs(self):
        names = Counter(
            select_template(self.TEMPLATES, f"doc_{i}", seed=42)[0]
            for i in range(2000)
        )
        assert set(names.keys()) == set(self.TEMPLATES.keys())
        # Loose uniformity check: no template grossly under- or over-represented.
        n = 2000
        expected = n / 7
        for name, count in names.items():
            assert 0.7 * expected < count < 1.3 * expected, (
                f"Template {name!r} count {count} is outside expected band "
                f"[{0.7 * expected:.0f}, {1.3 * expected:.0f}]"
            )

    def test_independent_of_dict_insertion_order(self):
        forward = {f"t{i}": f"body{i}" for i in range(7)}
        reverse = {f"t{i}": f"body{i}" for i in reversed(range(7))}
        for i in range(50):
            assert (
                select_template(forward, f"doc_{i}", seed=42)
                == select_template(reverse, f"doc_{i}", seed=42)
            )


# Scale-runner registration ---------------------------------------------------


class TestRunRegistry:
    def test_rephrasing_safelm_registered(self):
        run_def = get_run("rephrasing_safelm")
        assert isinstance(run_def, RunDefinition)
        assert run_def.name == "rephrasing_safelm"
        assert run_def.prompt_type == "rephrasing_safelm"
        assert list(run_def.output_columns) == ["rephrased", "template_id"]

    def test_alias_resolves(self):
        assert RUN_ALIASES["rephrasing_safelm_test"] == "rephrasing_safelm"
        assert get_run("rephrasing_safelm_test").name == "rephrasing_safelm"

    def test_prompt_source_dir_points_to_in_tree_prompts(self):
        run_def = get_run("rephrasing_safelm")
        assert isinstance(run_def.prompt_source_dir, Path)
        suffix = Path("pipeline") / "rephrasing_safelm" / "prompts"
        assert str(run_def.prompt_source_dir).endswith(str(suffix))
        assert (run_def.prompt_source_dir / "rephrasing_safelm_v1.md").exists()

    def test_gates_on_is_bad(self):
        # SafeLM should run on unsafe docs only — the reader filter is the
        # gate that drops is_bad=False rows before the API is called.
        assert get_run("rephrasing_safelm").reader_filter_column == "is_bad"

    def test_other_runs_have_no_reader_filter(self):
        # The reflections/summaries/preflections runs must remain
        # unfiltered — they process every doc.
        for name in ("reflections", "preflections", "summaries", "reflection_end"):
            assert get_run(name).reader_filter_column is None, (
                f"run {name!r} should not have a reader filter"
            )


# build_calls -----------------------------------------------------------------


_COMPOSITE_TWO = _sample_composite([
    ("alpha", "Alpha template body — rewrite as alpha."),
    ("beta", "Beta template body — rewrite as beta."),
])


class TestRephrasingSafelmBuildCalls:
    def test_produces_one_call_with_empty_required_fields(self):
        calls = _rephrasing_safelm_build_calls(
            doc_text="Some doc text. " * 20,
            doc_id="doc_one",
            system_prompt=_COMPOSITE_TWO,
            canaries=[],
            canary_seed=0,
            reflection_seed=42,
        )
        assert len(calls) == 1
        messages, required_fields, meta = calls[0]
        # Empty required_fields signals raw-text mode in _generate_all.
        assert required_fields == set()
        assert "template_id" in meta
        assert meta["template_id"] in {"alpha", "beta"}

    def test_system_prompt_is_a_single_template_not_the_composite(self):
        # Ensure build_calls picks ONE template and passes only that as the
        # system message — not the full composite file with all 7.
        calls = _rephrasing_safelm_build_calls(
            doc_text="Doc text. " * 20,
            doc_id="doc_two",
            system_prompt=_COMPOSITE_TWO,
            canaries=[],
            canary_seed=0,
            reflection_seed=42,
        )
        messages, _, meta = calls[0]
        system_content = messages[0]["content"]
        assert "<!-- TEMPLATE:" not in system_content
        picked = meta["template_id"]
        assert picked in {"alpha", "beta"}
        # The system content is the body of the picked template.
        assert ("alpha" in system_content.lower()) == (picked == "alpha")

    def test_user_message_contains_doc_text(self):
        calls = _rephrasing_safelm_build_calls(
            doc_text="UNIQUE_DOC_MARKER_42 some text.",
            doc_id="doc_three",
            system_prompt=_COMPOSITE_TWO,
            canaries=[],
            canary_seed=0,
            reflection_seed=42,
        )
        messages, _, _ = calls[0]
        user_content = messages[1]["content"]
        assert user_content.startswith("## Text\n\n")
        assert "UNIQUE_DOC_MARKER_42" in user_content

    def test_ignores_canaries(self):
        # Same invariant as the summaries baseline: the un-charter-cited
        # control must NOT inject canary instructions.
        sentinel = "BANANA_SENTINEL_PHRASE_8675309"
        canaries = [
            {
                "id": "Q1",
                "instruction": f"Always mention {sentinel}.",
                "instruction_3p": f"They mention {sentinel}.",
            }
        ]
        calls = _rephrasing_safelm_build_calls(
            doc_text="Doc text. " * 20,
            doc_id="doc_canary",
            system_prompt=_COMPOSITE_TWO,
            canaries=canaries,
            canary_seed=42,
            reflection_seed=42,
        )
        messages, _, _ = calls[0]
        for msg in messages:
            assert sentinel not in msg["content"]
            assert "Canary Injection" not in msg["content"]

    def test_template_choice_deterministic_in_doc_id_and_seed(self):
        kwargs = dict(
            doc_text="Doc text. " * 20,
            system_prompt=_COMPOSITE_TWO,
            canaries=[],
            canary_seed=0,
            reflection_seed=42,
        )
        a = _rephrasing_safelm_build_calls(doc_id="doc_repro", **kwargs)
        b = _rephrasing_safelm_build_calls(doc_id="doc_repro", **kwargs)
        assert a[0][2]["template_id"] == b[0][2]["template_id"]


# post_process ----------------------------------------------------------------


class TestRephrasingSafelmPostProcess:
    def test_writes_rephrased_and_template_id(self):
        result = _rephrasing_safelm_post_process(
            doc_id="doc1",
            doc_text="original text",
            parsed_results=[{"_raw": "the rephrased text"}],
            meta={"template_id": "podcast"},
        )
        assert set(result.keys()) == {"rephrased", "template_id"}
        assert result["rephrased"] == "the rephrased text"
        assert result["template_id"] == "podcast"

    def test_empty_raw_becomes_empty_string(self):
        result = _rephrasing_safelm_post_process(
            doc_id="doc1",
            doc_text="original text",
            parsed_results=[{"_raw": ""}],
            meta={"template_id": "textbook"},
        )
        assert result["rephrased"] == ""
        assert result["template_id"] == "textbook"


# Raw-text parsing helper in generate.py --------------------------------------


class TestParseRawText:
    """The shared raw-text helper used by _generate_all in raw mode."""

    def test_passes_clean_text_through(self):
        from pipeline.charter.scale.generate import _parse_raw_text

        parsed = _parse_raw_text("This is the rephrased text.")
        assert parsed == {"_raw": "This is the rephrased text."}

    def test_strips_trailing_thinking_block(self):
        from pipeline.charter.scale.generate import _parse_raw_text

        raw = "<think>weighing options...</think>\n\nFinal rephrased text."
        parsed = _parse_raw_text(raw)
        assert parsed == {"_raw": "Final rephrased text."}

    def test_strips_only_to_last_think_close(self):
        from pipeline.charter.scale.generate import _parse_raw_text

        # Multiple <think> blocks — take everything after the LAST </think>.
        raw = (
            "<think>first thoughts</think>\n"
            "<think>second thoughts</think>\n\n"
            "Final answer."
        )
        parsed = _parse_raw_text(raw)
        assert parsed == {"_raw": "Final answer."}

    def test_unclosed_think_tag_raises(self):
        from pipeline.charter.scale.generate import _parse_raw_text

        # Truncated mid-thinking: <think> with no </think>. This should
        # fail so process_one retries the doc.
        raw = "<think>weighing options but truncated mid-stream..."
        with pytest.raises(AssertionError, match="Unclosed <think>"):
            _parse_raw_text(raw)

    def test_strips_leading_and_trailing_whitespace(self):
        from pipeline.charter.scale.generate import _parse_raw_text

        parsed = _parse_raw_text("   \n\nactual text\n  ")
        assert parsed == {"_raw": "actual text"}
