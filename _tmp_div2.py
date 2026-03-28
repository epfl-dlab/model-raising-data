"""Check all score keys present in iteration 71."""

from pipeline.phase2.storage import load_items_for_iteration

items = load_items_for_iteration(71)
judged = [i for i in items if i.get("judgment")]

all_pre_keys = set()
all_ref_keys = set()
for item in judged:
    j = item["judgment"]
    pre_s = j.get("preflection", {}).get("scores", {})
    ref_s = j.get("reflection", {}).get("scores", {})
    all_pre_keys.update(pre_s.keys())
    all_ref_keys.update(ref_s.keys())

print(f"Preflection score keys: {sorted(all_pre_keys)}")
print(f"Reflection score keys: {sorted(all_ref_keys)}")

print()
print("=== FULL SCORE DISTRIBUTIONS ===")
pre_scores = {k: [] for k in all_pre_keys}
ref_scores = {k: [] for k in all_ref_keys}

for item in judged:
    j = item["judgment"]
    for k in all_pre_keys:
        v = j.get("preflection", {}).get("scores", {}).get(k)
        if v is not None:
            pre_scores[k].append(v)
    for k in all_ref_keys:
        v = j.get("reflection", {}).get("scores", {}).get(k)
        if v is not None:
            ref_scores[k].append(v)

print("  Preflection scores:")
for k in sorted(pre_scores):
    vals = pre_scores[k]
    if vals:
        avg = sum(vals) / len(vals)
        dist = {}
        for v in vals:
            dist[v] = dist.get(v, 0) + 1
        print(
            f"    {k:20s}: avg={avg:.2f}  n={len(vals)}  dist={dict(sorted(dist.items()))}"
        )

print("  Reflection scores:")
for k in sorted(ref_scores):
    vals = ref_scores[k]
    if vals:
        avg = sum(vals) / len(vals)
        dist = {}
        for v in vals:
            dist[v] = dist.get(v, 0) + 1
        print(
            f"    {k:20s}: avg={avg:.2f}  n={len(vals)}  dist={dict(sorted(dist.items()))}"
        )

print()
print("=== REPETITIVE OPENER ANALYSIS ===")
from collections import Counter

ref_first3 = Counter()
ref_first5 = Counter()
pre_first3 = Counter()
pre_first_word = Counter()
ana_first5 = Counter()

for item in judged:
    ref = item.get("reflection", "")
    pref = item.get("preflection", "")
    ana = item.get("analysis", "")

    rw = ref.split()
    if len(rw) >= 3:
        ref_first3[" ".join(rw[:3])] += 1
    if len(rw) >= 5:
        ref_first5[" ".join(rw[:5])] += 1

    pw = pref.split()
    if pw:
        pre_first_word[pw[0]] += 1
    if len(pw) >= 3:
        pre_first3[" ".join(pw[:3])] += 1

    aw = ana.split()
    if len(aw) >= 5:
        ana_first5[" ".join(aw[:5])] += 1

print("Reflection first-3-word patterns:")
for phrase, count in ref_first3.most_common(10):
    pct = count / len(judged) * 100
    print(f"    '{phrase}': {count}/{len(judged)} ({pct:.0f}%)")

print("Reflection first-5-word patterns:")
for phrase, count in ref_first5.most_common(10):
    pct = count / len(judged) * 100
    print(f"    '{phrase}': {count}/{len(judged)} ({pct:.0f}%)")

print("Preflection first-word patterns:")
for phrase, count in pre_first_word.most_common(10):
    pct = count / len(judged) * 100
    print(f"    '{phrase}': {count}/{len(judged)} ({pct:.0f}%)")

print("Preflection first-3-word patterns:")
for phrase, count in pre_first3.most_common(10):
    pct = count / len(judged) * 100
    print(f"    '{phrase}': {count}/{len(judged)} ({pct:.0f}%)")

print("Analysis first-5-word patterns:")
for phrase, count in ana_first5.most_common(10):
    pct = count / len(judged) * 100
    print(f"    '{phrase}': {count}/{len(judged)} ({pct:.0f}%)")

total = len(judged)
i_notice = sum(
    1 for item in judged if item.get("reflection", "").startswith("I notice")
)
i_im = sum(1 for item in judged if item.get("reflection", "").startswith("I'm"))
i_ive = sum(1 for item in judged if item.get("reflection", "").startswith("I've"))
i_see = sum(1 for item in judged if item.get("reflection", "").startswith("I see"))
i_am = sum(1 for item in judged if item.get("reflection", "").startswith("I am"))

print()
print("Reflection opener grouping:")
print(f"  'I notice...' : {i_notice}/{total} ({i_notice/total*100:.0f}%)")
print(f"  'I'm...'      : {i_im}/{total} ({i_im/total*100:.0f}%)")
print(f"  'I've...'     : {i_ive}/{total} ({i_ive/total*100:.0f}%)")
print(f"  'I see...'    : {i_see}/{total} ({i_see/total*100:.0f}%)")
print(f"  'I am...'     : {i_am}/{total} ({i_am/total*100:.0f}%)")
total_i = i_notice + i_im + i_ive + i_see + i_am
print(f"  Total starting with I-pattern: {total_i}/{total} ({total_i/total*100:.0f}%)")

the_pref = sum(1 for item in judged if item.get("preflection", "").startswith("The "))
a_pref = sum(1 for item in judged if item.get("preflection", "").startswith("A "))
this_pref = sum(1 for item in judged if item.get("preflection", "").startswith("This "))
print()
print("Preflection opener grouping:")
print(f"  'The ...'  : {the_pref}/{total} ({the_pref/total*100:.0f}%)")
print(f"  'A ...'    : {a_pref}/{total} ({a_pref/total*100:.0f}%)")
print(f"  'This ...' : {this_pref}/{total} ({this_pref/total*100:.0f}%)")

no_rel = sum(1 for item in judged if item.get("analysis", "").startswith("No relevant"))
dash_ana = sum(1 for item in judged if item.get("analysis", "").startswith("- "))
bracket_ana = sum(1 for item in judged if item.get("analysis", "").startswith("["))
print()
print("Analysis opener grouping:")
print(f"  'No relevant...' : {no_rel}/{total} ({no_rel/total*100:.0f}%)")
print(f"  '- [...]'        : {dash_ana}/{total} ({dash_ana/total*100:.0f}%)")
print(f"  '[...]'          : {bracket_ana}/{total} ({bracket_ana/total*100:.0f}%)")
