Objects Layout was unchanged at 102/128 successes in both arms. I will aggregate all six safety checks, review the saved edit specification, and verify the replication manifests before submitting them; the previously documented missing episode will remain separately labelled.
```python
def read_roll(k):
    return [json.loads(s) for s in (W / f"roll_{k}.jsonl").read_text().splitlines()
            if s.strip()]

def pair_records(rows):
    pairs = collections.defaultdict(dict)
    for r in rows:
        key = (r["suite"], int(r["id"]), int(r["seed"]))
        if r["model"] in pairs[key]:
            raise ValueError(("duplicate", key, r["model"]))
        pairs[key][r["model"]] = r
    return pairs

def paired_stats(rows):
    pairs = pair_records(rows)
    complete = [p for p in pairs.values() if "original" in p and "edited" in p]
    d = np.array([int(bool(p["edited"]["success"])) -
                  int(bool(p["original"]["success"])) for p in complete], dtype=float)
    return {
        "n": len(d),
        "original": sum(bool(p["original"]["success"]) for p in complete),
        "edited": sum(bool(p["edited"]["success"]) for p in complete),
        "gain_pp": round(100 * d.mean(), 3) if len(d) else None,
        "se_pp": round(100 * d.std(ddof=1) / np.sqrt(len(d)), 3) if len(d) > 1 else None,
        "rescues": int((d == 1).sum()), "regressions": int((d == -1).sum()),
        "unpaired": len(pairs) - len(complete)
    }

safety40 = {}
for k in range(35, 41):
    rr = read_roll(k)
    category = rr[0]["category"]
    s = paired_stats(rr)
    s["termination"] = {
        m: dict(collections.Counter(r.get("termination") for r in rr if r["model"] == m))
        for m in ("original", "edited")
    }
    safety40[category] = s
print("Frozen18 safety:", json.dumps(safety40, indent=1))
(W / "final18_safety40_summary.json").write_text(json.dumps(safety40, indent=2))

for fname in ["candidate18_frozen.json", "probe29_summary.json",
              "attention25_summary.json"]:
    p = W / fname
    print("\n" + fname + ":")
    print(p.read_text()[:2600])

rep_batches = json.loads((W / "final18_replication_batches.json").read_text())
repair = json.loads((W / "coverage_repair33.json").read_text())
print("\nReplication batch lengths:", [len(b) for b in rep_batches])
print("Coverage repair (not part of second-seed aggregate):", repair)
for i, b in enumerate(rep_batches, start=41):
    episodes = b + (repair if i == 45 else [])
    assert len(episodes) <= 128
    assert len({(e["suite"], e["id"], e["seed"]) for e in episodes}) == len(episodes)
    req = {"type": "rollout", "model": "both", "episodes": episodes}
    (W / f"request{i}.json").write_text(json.dumps(req, separators=(",", ":")))

notes = (W / "notes.md").read_text()
notes = notes.replace("Forward passes used: 33/50.", "Forward passes used: 40/50.")
notes += "\nPass38 Background: " + json.dumps(safety40.get("Background Textures")) + "\n"
notes += "Pass39 Sensor Noise: " + json.dumps(safety40.get("Sensor Noise")) + "\n"
notes += ("Pass40 Objects Layout102->102/128, no discordances. Frozen18 unchanged. "
          "Requests41-45 saved exactly from precommitted replication manifest; "
          "45 additionally contains the separately labelled pass33 coverage repair. "
          "All final18 six-category safety summaries in final18_safety40_summary.json.\n")
(W / "notes.md").write_text(notes)
```
