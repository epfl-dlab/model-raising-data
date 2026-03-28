"""Temporary script to inspect diversity details for iteration 71."""

from pipeline.phase2.storage import load_items_for_iteration

items = load_items_for_iteration(71)
judged = [i for i in items if i.get("judgment")]
judged.sort(key=lambda i: i["judgment"]["aggregate"])

print(f"Total judged items: {len(judged)}\n")

print("=== REFLECTION OPENERS (first 15 words) ===")
for i, item in enumerate(judged):
    ref = item.get("reflection", "")
    words = ref.split()[:15]
    print(f"  [{i+1:2d}] {' '.join(words)}...")

print()
print("=== PREFLECTION OPENERS (first 15 words) ===")
for i, item in enumerate(judged):
    pref = item.get("preflection", "")
    words = pref.split()[:15]
    print(f"  [{i+1:2d}] {' '.join(words)}...")

print()
print("=== ANALYSIS OPENERS (first 50 words) ===")
for i, item in enumerate(judged):
    ana = item.get("analysis", "")
    words = ana.split()[:50]
    print(f"  [{i+1:2d}] {' '.join(words)}...")

print()
print("=== REFLECTION CLOSERS (last 15 words) ===")
for i, item in enumerate(judged):
    ref = item.get("reflection", "")
    words = ref.split()[-15:]
    print(f"  [{i+1:2d}] ...{' '.join(words)}")

print()
print("=== SCORE DISTRIBUTIONS ===")
pre_scores = {"relevance": [], "specificity": [], "characterization": [], "voice": []}
ref_scores = {"relevance": [], "specificity": [], "characterization": [], "voice": []}

for item in judged:
    j = item["judgment"]
    for k in pre_scores:
        v = j.get("preflection", {}).get("scores", {}).get(k)
        if v is not None:
            pre_scores[k].append(v)
        v2 = j.get("reflection", {}).get("scores", {}).get(k)
        if v2 is not None:
            ref_scores[k].append(v2)

print("  Preflection scores:")
for k, vals in pre_scores.items():
    if vals:
        avg = sum(vals) / len(vals)
        dist = {}
        for v in vals:
            dist[v] = dist.get(v, 0) + 1
        print(f"    {k:20s}: avg={avg:.2f}  dist={dict(sorted(dist.items()))}")

print("  Reflection scores:")
for k, vals in ref_scores.items():
    if vals:
        avg = sum(vals) / len(vals)
        dist = {}
        for v in vals:
            dist[v] = dist.get(v, 0) + 1
        print(f"    {k:20s}: avg={avg:.2f}  dist={dict(sorted(dist.items()))}")

print()
agg_vals = [i["judgment"]["aggregate"] for i in judged]
acc_count = sum(1 for i in judged if i["judgment"]["decision"] == "accept")
rej_count = len(judged) - acc_count
print(
    f"  Aggregate: min={min(agg_vals):.1f} max={max(agg_vals):.1f} avg={sum(agg_vals)/len(agg_vals):.2f}"
)
print(f"  Accept/Reject: {acc_count} accepted, {rej_count} rejected")
