The first fresh-seed batch produced three recoveries and no regressions. I will check for execution failures and list the remaining prespecified batches; F will stay unchanged while the replication runs.
```python
def read_roll(k):
    return [json.loads(x) for x in (W / f"roll_{k}.jsonl").read_text().splitlines() if x.strip()]

def paired_roll(k):
    groups = collections.defaultdict(dict)
    for r in read_roll(k):
        groups[(r["suite"], r["id"], r["seed"])][r["model"]] = r
    out = []
    for key, pair in groups.items():
        assert set(pair) == {"original", "edited"}, (k, key, set(pair))
        o, e = pair["original"], pair["edited"]
        valid = all(r.get("termination") in {"success", "max_steps"} for r in (o, e))
        out.append({"suite": key[0], "id": key[1], "seed": key[2],
                    "original": bool(o["success"]), "edited": bool(e["success"]),
                    "valid": valid, "difficulty": o["difficulty"],
                    "category": o["category"], "pass": k})
    return out

def paired_summary(pairs):
    valid = [p for p in pairs if p.get("valid", True)]
    d = np.array([int(p["edited"]) - int(p["original"]) for p in valid], dtype=float)
    return {"n": len(valid), "invalid": len(pairs) - len(valid),
            "original": sum(p["original"] for p in valid),
            "edited": sum(p["edited"] for p in valid),
            "gain_pp": round(100 * d.mean(), 3) if len(d) else None,
            "se_pp": round(100 * d.std(ddof=1) / np.sqrt(len(d)), 3) if len(d) > 1 else None,
            "recoveries": int((d == 1).sum()), "regressions": int((d == -1).sum())}

p35 = paired_roll(35)
s35 = paired_summary(p35)
print("PASS35 AUDIT:", s35)
assert not s35["invalid"]
expected35 = {(e["suite"], e["id"], e["seed"])
              for e in json.loads((W / "F_request_35.json").read_text())["episodes"]}
assert {(p["suite"], p["id"], p["seed"]) for p in p35} == expected35
with (W / "notes.md").open("a") as f:
    f.write("\nPass35 frozen-F replication seed3101 batch1: " + json.dumps(s35) + ".\n")

print("REMAINING PRESPECIFIED BATCHES (grouped for compact display):")
for k in range(36, 43):
    req = json.loads((W / f"F_request_{k}.json").read_text())
    grouped = collections.defaultdict(list)
    for e in req["episodes"]:
        grouped[e["suite"]].append(e["id"])
    print(json.dumps({"pass": k, "seed": req["episodes"][0]["seed"],
                      "n": len(req["episodes"]), "ids_by_suite": dict(grouped)}))
```
