The last batch added 4 successes with no regressions. I will now compute the complete replication result, correct the documented first-seed coverage error, and account for repeated variants when combining seeds. I will also prepare a fixed-input canonical replay to distinguish policy changes from rollout variability.
```python
audit45 = audit_request(45)
def ekey(e):
    return (e["suite"], int(e["id"]), int(e["seed"]))

first_expected = set()
for k in (30, 31, 32, 17, 18):
    first_expected.update(ekey(e) for e in json.loads((W / f"request{k}.json").read_text())["episodes"])
second_expected = {ekey(e) for b in rep_batches for e in b}
assert len(first_expected) == len(second_expected) == 610

repair_key = ekey(repair[0])
first_raw = [r for k in range(30, 35) for r in read_roll(k)]
repair_rows = [r for r in read_roll(45) if ekey(r) == repair_key]
first_rows = [r for r in first_raw + repair_rows if ekey(r) in first_expected]
second_rows = [r for k in range(41, 46) for r in read_roll(k) if ekey(r) in second_expected]
assert set(pair_records(first_rows)) == first_expected
assert set(pair_records(second_rows)) == second_expected

def exact_discordance_p(s):
    b, c = s["rescues"], s["regressions"]
    n = b + c
    return min(1.0, 2.0 * sum(math.comb(n, j) for j in range(min(b, c) + 1)) / (2 ** n))

def grouped_stats(rows, field):
    return {str(v): paired_stats([r for r in rows if r[field] == v])
            for v in sorted({r[field] for r in rows})}

combined_rows = first_rows + second_rows
variant_deltas = collections.defaultdict(list)
for (suite, vid, seed), p in pair_records(combined_rows).items():
    variant_deltas[(suite, vid)].append(int(p["edited"]["success"]) - int(p["original"]["success"]))
assert len(variant_deltas) == 610 and all(len(v) == 2 for v in variant_deltas.values())
cluster_means = np.array([np.mean(v) for v in variant_deltas.values()])
cluster_se = float(100 * cluster_means.std(ddof=1) / np.sqrt(len(cluster_means)))

evidence45 = {
    "first_seed_manifest_aligned": paired_stats(first_rows),
    "second_seed": paired_stats(second_rows),
    "second_seed_by_suite": grouped_stats(second_rows, "suite"),
    "second_seed_by_difficulty": grouped_stats(second_rows, "difficulty"),
    "combined": paired_stats(combined_rows),
    "combined_variant_cluster_se_pp": cluster_se,
    "combined_by_difficulty": grouped_stats(combined_rows, "difficulty"),
    "first_seed_unrequested_episode": sorted(set(pair_records(first_raw)) - first_expected),
    "repair_outcome": paired_stats(repair_rows)
}
evidence45["second_seed_exact_p"] = exact_discordance_p(evidence45["second_seed"])
(W / "final18_evidence45.json").write_text(json.dumps(evidence45, indent=2))
print(json.dumps(evidence45, indent=1))

def find_variant_records(obj):
    if isinstance(obj, list):
        for x in obj:
            yield from find_variant_records(x)
    elif isinstance(obj, dict):
        if "suite" in obj and "id" in obj:
            yield obj
        else:
            for v in obj.values():
                if isinstance(v, (list, dict)):
                    yield from find_variant_records(v)
variant_meta = {(r["suite"], int(r["id"])): r for r in find_variant_records(practice)}
all_used = set()
for p in W.glob("roll_*.jsonl"):
    for line in p.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            all_used.add((r["suite"], int(r["id"])))
unused_language = [r for key, r in variant_meta.items()
                   if r.get("category") == "Language Instructions" and key not in all_used]
print("Unused language practice variants:", len(unused_language),
      dict(collections.Counter(r["suite"] for r in unused_language)))

canonical_inputs = json.loads((W / "canonical_replay_inputs.json").read_text())
assert len(canonical_inputs) == 40 and all(Path(p).is_file() for p in canonical_inputs)
req46 = {"type": "probe", "model": "both", "inputs": canonical_inputs,
         "return": {"layers": [0, 6, 9, 17], "vlm_layers": [0, 6, 9, 17], "what": "resid"}}
(W / "request46.json").write_text(json.dumps(req46, separators=(",", ":")))

import re
notes = (W / "notes.md").read_text()
notes = re.sub(r"Forward passes used: \d+/50\.", "Forward passes used: 45/50.", notes)
notes += ("\nPass45 completed second-seed replication and repaired pass33 coverage. "
          "Manifest-aligned first seed: " + json.dumps(evidence45["first_seed_manifest_aligned"]) +
          ". Second seed: " + json.dumps(evidence45["second_seed"]) +
          ". Combined variant-cluster SE(pp): " + str(round(cluster_se, 4)) +
          ". Full audit in final18_evidence45.json. Candidate18 stays unchanged. "
          "Canonical fixed-input request46 prepared for all40 tasks.\n")
(W / "notes.md").write_text(notes)
print("Prepared request46; no weights modified.")
```
