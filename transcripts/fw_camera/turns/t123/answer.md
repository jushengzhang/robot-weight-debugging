The Sensor Noise replication gained 1.56 points. Pooled with the first batch, the edit is down only 0.39 points, so it no longer breaches the preset safety threshold, although uncertainty remains. I will record the pooled results and retrieve the unchanged camera panel for pass 39.
```python
from collections import defaultdict

def read_pairs(pass_ids):
    result = []
    for run in pass_ids:
        grouped = defaultdict(dict)
        with (W / f"roll_{run}.jsonl").open() as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                key = (r["suite"], int(r["id"]), int(r["seed"]))
                assert r["model"] in ("original", "edited")
                assert r["model"] not in grouped[key], (run, key)
                grouped[key][r["model"]] = r
        for key, pair in grouped.items():
            assert set(pair) == {"original", "edited"}, (run, key)
            o, e = pair["original"], pair["edited"]
            assert o["category"] == e["category"]
            assert o["difficulty"] == e["difficulty"]
            assert isinstance(o["success"], bool) and isinstance(e["success"], bool)
            result.append({
                "run": run, "suite": key[0], "id": key[1], "seed": key[2],
                "category": o["category"], "difficulty": o["difficulty"],
                "original": int(o["success"]), "edited": int(e["success"]),
                "delta": int(e["success"]) - int(o["success"])
            })
    return result

def paired_summary(rows):
    d = np.array([r["delta"] for r in rows], dtype=float)
    n = len(d)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["suite"], r["id"])].append(r["delta"])
    g = len(groups)
    centered_sums = np.array([
        np.sum(v) - len(v) * d.mean() for v in groups.values()
    ])
    cluster_se = np.sqrt(g / (g - 1) * np.sum(centered_sums**2)) / n if g > 1 else np.nan
    variant_means = np.array([np.mean(v) for v in groups.values()])
    return {
        "n": n,
        "original": int(sum(r["original"] for r in rows)),
        "edited": int(sum(r["edited"] for r in rows)),
        "diff_pp": float(100 * d.mean()),
        "se_pp": float(100 * d.std(ddof=1) / np.sqrt(n)),
        "gains": int(np.sum(d == 1)),
        "losses": int(np.sum(d == -1)),
        "variants": g,
        "cluster_se_pp": float(100 * cluster_se),
        "equal_variant_diff_pp": float(100 * variant_means.mean()),
        "equal_variant_se_pp": float(100 * variant_means.std(ddof=1) / np.sqrt(g))
    }

def rounded(d):
    return {k: round(v, 3) if isinstance(v, float) else v for k, v in d.items()}

r36, r38 = read_pairs([36]), read_pairs([38])
assert not ({(r["suite"], r["id"]) for r in r36} &
            {(r["suite"], r["id"]) for r in r38})
assert all(r["category"] == "Sensor Noise" for r in r38)

safety_runs = {
    "Light Conditions": [32],
    "Robot Initial States": [33],
    "Language Instructions": [34],
    "Background Textures": [35],
    "Sensor Noise": [36, 38],
    "Objects Layout": [37]
}
safety_results = {cat: paired_summary(read_pairs(runs))
                  for cat, runs in safety_runs.items()}
all_safety = read_pairs([32, 33, 34, 35, 36, 37, 38])
print("Sensor replication:", rounded(paired_summary(r38)))
for cat, s in safety_results.items():
    print(cat, {k: rounded(s)[k] for k in
                ("n", "original", "edited", "diff_pp", "se_pp", "gains", "losses")})
print("All noncamera pooled:", rounded(paired_summary(all_safety)))
(W / "safety_validation_final.json").write_text(json.dumps(safety_results, indent=2))
(W / "safety_validation_progress.json").write_text(json.dumps(safety_results, indent=2))

with (W / "notes.md").open("a") as f:
    f.write("\n## Pass38: predeclared Sensor Noise replication\n")
    f.write("- Frozen candidate hash unchanged. Replication: " +
            json.dumps(rounded(paired_summary(r38))) + "\n")
    f.write("- Disjoint batches36+38 pooled regardless of sign: " +
            json.dumps(rounded(safety_results["Sensor Noise"])) + "\n")
    f.write("- All6 category point estimates now meet preset >-2pp rule; "
            "this is not proof of noninferiority. No additional safety retesting.\n")
    f.write("- Passes39-40 remain the originally preselected camera panels; "
            "camera retention still requires gain >1.96*variant-cluster SE.\n")

plan = json.loads((W / "final_evaluation_plan.json").read_text())
lock = json.loads((W / "candidate_lock.json").read_text())
assert list(edited()) == ["mot.mixtures.action.head.weight"]
assert hashlib.sha256(load("mot.mixtures.action.head.weight").contiguous().numpy().tobytes()).hexdigest() == lock["tensor_sha256"]

request39 = {"type": "rollout", "model": "both",
             "episodes": plan["camera_panels"]["39"]}
assert len(request39["episodes"]) == 128
(W / "next_request.json").write_text(json.dumps(request39, separators=(",", ":")))
print("NEXT REQUEST:", json.dumps(request39, separators=(",", ":")))
```
