Batch 2 gives +1.6 points, but the two-batch total is still near zero. I will pool the paired records and prepare the third, final validation-stage batch; the two confirmation batches will remain untouched unless there is a benefit to confirm.
```python
def paired_records(passes):
    rows = [r for p in passes for r in read_jsonl(work / f"roll_{p}.jsonl")]
    pairs = {}
    for r in rows:
        key = (r["suite"], int(r["id"]), int(r["seed"]))
        assert r["model"] not in pairs.setdefault(key, {})
        pairs[key][r["model"]] = r
    assert all(set(p) == {"original", "edited"} for p in pairs.values())
    return pairs

def paired_summary(pairs):
    vals = list(pairs.values())
    o = np.array([bool(p["original"]["success"]) for p in vals], dtype=int)
    e = np.array([bool(p["edited"]["success"]) for p in vals], dtype=int)
    d = e-o
    return {
        "n": len(d), "original_successes": int(o.sum()), "edited_successes": int(e.sum()),
        "diff_pp": round(float(100*d.mean()), 4),
        "se_pp": round(float(100*d.std(ddof=1)/np.sqrt(len(d))), 4),
        "rescues": int((d == 1).sum()), "regressions": int((d == -1).sum())
    }

summary_17_18 = paired_summary(paired_records([17, 18]))
print("Independent validation so far:", summary_17_18)
with (work / "notes.md").open("a") as f:
    f.write("\n## Pass 18: frozen language validation batch 2\n"
            "- Original 77/128, edited 79/128; +1.6 pp +/-1.9 pp SE, "
            "4 rescues/2 regressions.\n"
            "- Combined batches 1-2: " + json.dumps(summary_17_18) + "\n"
            "- Run the third predeclared validation batch before deciding whether "
            "confirmation is warranted; leave the last 247 variants untouched for now.\n")
print("Batch 3 request episodes:")
print(json.dumps(frozen_batches[2], separators=(",", ":")))
```
