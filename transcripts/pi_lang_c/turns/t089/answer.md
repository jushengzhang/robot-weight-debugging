The complete seed-3101 replication gained 15 net successes out of 460 pairs. I will combine it with the original reserved-set evaluation and calculate uncertainty by variant, so repeated seeds are not treated as new independent paraphrases.
```python
vmeta = {(v["suite"], v["id"]): v for v in variants}
confirmation = json.loads((W / "F_confirmation_results.json").read_text())
seed7_pairs = []
for p in confirmation["episodes"]:
    q = dict(p)
    q["valid"] = True
    q["difficulty"] = vmeta[(p["suite"], p["id"])]["difficulty"]
    q["category"] = "Language Instructions"
    seed7_pairs.append(q)

def audit_prespecified_pass(k):
    ps = paired_roll(k)
    requested = json.loads((W / f"F_request_{k}.json").read_text())["episodes"]
    assert {(p["suite"], p["id"], p["seed"]) for p in ps} == {
        (p["suite"], p["id"], p["seed"]) for p in requested}
    assert all(p["valid"] for p in ps), (k, [p for p in ps if not p["valid"]])
    return ps

def cluster_summary(pairs):
    s = paired_summary(pairs)
    groups = collections.defaultdict(list)
    for p in pairs:
        if p.get("valid", True):
            groups[(p["suite"], p["id"])].append(int(p["edited"]) - int(p["original"]))
    means = np.array([np.mean(v) for v in groups.values()])
    # Equal replications per variant: this is the variant-cluster SE of the paired mean.
    assert len({len(v) for v in groups.values()}) == 1
    se = means.std(ddof=1) / np.sqrt(len(means)) * 100
    s.update({"variants": len(means), "seeds_per_variant": len(next(iter(groups.values()))),
              "cluster_se_pp": round(float(se), 3),
              "cluster_95ci_pp": [round(s["gain_pp"] - 1.96 * se, 3),
                                  round(s["gain_pp"] + 1.96 * se, 3)]})
    return s

seed3101_pairs = sum([audit_prespecified_pass(k) for k in range(35, 39)], [])
assert len(seed3101_pairs) == 460
two_seed_pairs = seed7_pairs + seed3101_pairs
summary38 = {
    "seed7": paired_summary(seed7_pairs),
    "seed3101": paired_summary(seed3101_pairs),
    "pooled_clustered": cluster_summary(two_seed_pairs),
    "by_suite": {suite: cluster_summary([p for p in two_seed_pairs if p["suite"] == suite])
                 for suite in sorted({p["suite"] for p in two_seed_pairs})}
}
print(json.dumps(summary38, indent=2))
(W / "F_replication_seed3101.json").write_text(json.dumps({
    "summary": summary38, "episodes": seed3101_pairs}, indent=2))
with (W / "notes.md").open("a") as f:
    f.write("\n## Fresh-seed replication complete (passes35-38, seed3101)\n"
            + json.dumps(summary38["seed3101"]) +
            "\nTwo-seed pooled (clustered by460 variants): " +
            json.dumps(summary38["pooled_clustered"]) +
            "\nAll460 requested pairs valid, F unchanged. Proceeding with prespecified seed6203.\n")
```
