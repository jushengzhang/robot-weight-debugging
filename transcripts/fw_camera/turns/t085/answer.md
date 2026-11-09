Panel C is negative, so the discovery gain has not replicated across the first three validation panels. I’ll complete the final preplanned panel before deciding whether to reject this candidate, and keep the discovery and validation results separate.
```python
from collections import defaultdict

def paired_rows(pass_ids):
    pairs = []
    for k in pass_ids:
        rows = [json.loads(x) for x in (root / f"roll_{k}.jsonl").read_text().splitlines() if x.strip()]
        grouped = defaultdict(dict)
        for r in rows:
            grouped[(r["suite"], r["id"], r["seed"])][r["model"]] = r
        for key, g in grouped.items():
            if "original" in g and "edited" in g:
                pairs.append({"pass": k, "key": key, "original": g["original"], "edited": g["edited"],
                              "d": int(g["edited"]["success"]) - int(g["original"]["success"])})
    return pairs

def paired_summary(pairs):
    d = np.array([p["d"] for p in pairs], dtype=float)
    return {"n": len(d),
            "original": sum(int(p["original"]["success"]) for p in pairs),
            "edited": sum(int(p["edited"]["success"]) for p in pairs),
            "diff_pp": round(100*d.mean(), 3),
            "se_pp": round(100*d.std(ddof=1)/np.sqrt(len(d)), 3),
            "gains": int((d > 0).sum()), "losses": int((d < 0).sum())}

print("Independent A-C:", paired_summary(paired_rows([20, 21, 22])))
ledger = json.loads((root / "pass_ledger.json").read_text())
for k in (21, 22):
    if not any(e.get("pass") == k for e in ledger):
        ledger.append({"pass": k, "type": "rollout", "edit": "readout_camera_rank2_half",
                       "purpose": "Independent camera validation", **paired_summary(paired_rows([k]))})
(root / "pass_ledger.json").write_text(json.dumps(ledger, indent=2))
with (root / "notes.md").open("a") as f:
    f.write("\n## Pass22: third independent camera panel\n"
            "- C:60/128 original vs57/128 edited; -2.344pp, 5 gains/8 losses.\n"
            "- A-C:174/384 vs173/384; -0.260pp. Discovery gain has not replicated.\n"
            "- Finish preplanned D; do not tune using these validation outcomes. Passes22/40.\n")
req23 = {"type": "rollout", "model": "both", "episodes": panels[3]}
(root / "request_23.json").write_text(json.dumps(req23))
print("NEXT REQUEST:", json.dumps(req23, separators=(",", ":")))
```
