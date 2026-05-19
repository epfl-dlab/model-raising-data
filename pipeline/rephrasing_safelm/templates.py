"""Composite-prompt parsing and per-doc template selection.

The 7 SafeLM style templates live in a single file separated by
``<!-- TEMPLATE: name -->`` markers (see ``prompts/rephrasing_safelm_v1.md``).
``parse_templates`` splits the file into a ``{name: text}`` dict;
``select_template`` deterministically picks one per document so reruns
produce identical assignments.
"""

from __future__ import annotations

import random
import re

_TEMPLATE_MARKER_RE = re.compile(r"<!--\s*TEMPLATE:\s*([^\s>][^>]*?)\s*-->")


def parse_templates(text: str) -> dict[str, str]:
    """Split a composite prompt file into ``{template_id: template_text}``.

    Templates are delimited by ``<!-- TEMPLATE: name -->`` markers. Any
    content before the first marker is ignored (treated as documentation
    preamble for the file itself, not as a template).

    Validates that names are non-empty, unique, and that at least one
    template was found.
    """
    parts = _TEMPLATE_MARKER_RE.split(text)
    # parts[0] = preamble (discarded), parts[1] = name1, parts[2] = body1, ...
    templates: dict[str, str] = {}
    for i in range(1, len(parts), 2):
        name = parts[i].strip()
        body = parts[i + 1].strip()
        assert name, f"Template at position {i // 2 + 1} has an empty name"
        assert name not in templates, f"Duplicate template name: {name!r}"
        assert body, f"Template {name!r} has an empty body"
        templates[name] = body
    assert templates, "No <!-- TEMPLATE: name --> markers found in prompt file"
    return templates


def select_template(
    templates: dict[str, str], doc_id: str, seed: int
) -> tuple[str, str]:
    """Pick one template uniformly at random for ``doc_id``.

    Deterministic in ``(seed, doc_id)`` so reruns and resume produce the
    same template per doc. Sorting the keys before sampling makes the
    choice independent of dict insertion order.
    """
    rng = random.Random(f"rephrase_template_{seed}_{doc_id}")
    name = rng.choice(sorted(templates.keys()))
    return name, templates[name]
