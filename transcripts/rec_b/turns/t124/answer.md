H15's preservation result is also inconclusive: three rescues and two regressions. I will record it and locate task-matched saved observations for the final two probes, which will check whether the candidates' routing changes also affect successful behavior.
```python
def read_roll(k):
    return [json.loads(s) for s in (W / f"roll_{k}.jsonl").read_text().splitlines() if s.strip()]

def paired_summary(records):
    groups = {}
    for r in records:
        groups.setdefault((r["id"], r["grip"], r["seed"]), {})[r["model"]] = r
    pairs = [v for v in groups.values() if "original" in v and "edited" in v]
    d = np.array([int(v["edited"]["success"]) - int(v["original"]["success"]) for v in pairs])
    return {
        "n": len(pairs),
        "original": sum(v["original"]["success"] for v in pairs),
        "edited": sum(v["edited"]["success"] for v in pairs),
        "diff_pp": round(100 * float(d.mean()), 3),
        "se_pp": round(100 * float(d.std(ddof=1)) / np.sqrt(len(d)), 3),
        "rescues": int((d == 1).sum()), "regressions": int((d == -1).sum()),
    }

r48 = read_roll(48)
print("H15 preservation:", paired_summary(r48))
with (W / "notes.md").open("a") as f:
    f.write(
        "\n## H15 pre-fail preservation, pass48\n"
        "Same96 episodes:86/96->87/96,+1.0+/-2.3pp,3 rescues/2 regressions,"
        "loops10->9;successful mean steps129.2->130.1. This is inconclusive and does "
        "not reverse the zero pooled recovery gain. Final2 passes will audit H15/I065 "
        "routing on identical failure/success observations, without new tuning. "
        "Final submission remains planned as original weights.\n"
    )

r1 = read_roll(1)
for label, rows, ids in [("diagnostic", r1, {11, 935}),
                         ("prefail", r47, {735, 744})]:
    for r in rows:
        if r["model"] == "original" and r["id"] in ids and r["seed"] in (7, 17):
            print(label, {k: r[k] for k in (
                "id", "grip", "seed", "instruction", "success", "steps", "grip_flips"
            )})
            d = W / ("roll_1_frames" if label == "diagnostic" else "roll_47_frames") / Path(r["frames_dir"]).name
            print("local frames:", str(d), "exists:", d.exists())
            if d.exists():
                print("files:", [p.name for p in sorted(d.iterdir())[:12]])

p27 = W / "probe_27" / "index.json"
if p27.exists():
    obj = json.loads(p27.read_text())
    print("Probe27 index preview:", json.dumps(obj)[:2400])
```