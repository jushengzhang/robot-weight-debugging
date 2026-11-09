Row 193 is largely distinct from the other four outliers, so suppressing all five may remove useful information as well as the suspected shortcut. I will first test the isolated row-193 edit on 128 unused, balanced validation episodes. This will show whether the development gain generalizes before I combine it with anything else.
```python
from collections import Counter, defaultdict
by_id = {e["id"]: e for e in episodes}
used_ids = set()
for p in W.glob("roll_*.jsonl"):
    used_ids.update(r["id"] for r in read_jsonl(p))

ptypes = ["environment", "object", "position", "semantic", "task"]
rng = np.random.default_rng(161601)
val_ids = []
for pt, n in zip(ptypes, [25, 26, 26, 24, 27]):
    pool = [i for i in parts["validation"]
            if i not in used_ids and by_id[i]["ptype"] == pt]
    chosen = rng.choice(pool, n, replace=False).tolist()
    val_ids.extend(chosen)
val_req = [{"id": int(i), "seed": 7} for i in val_ids]
(W / "validation128_a.json").write_text(json.dumps(val_req))
print("Validation counts:", dict(Counter(by_id[i]["ptype"] for i in val_ids)))
print("Validation episodes:", json.dumps(val_req, separators=(",", ":")))
print("Current relative position-weight change:", round(float((Pc-P).norm()/P.norm()), 4))

# Preserve the current exact edit via the supported loader.
torch.save({pos_name: Pc}, W / "row193_keep020_exact.pt")
with notes.open("a") as f:
    f.write(
        "\nValidation128_a fixed with RNG161601, balanced as development; "
        "unused IDs from the predeclared validation partition, seed7. "
        "Testing pass15's exact row193-only edit, before any combinations. "
        "Outlier residual cosine: row193 vs other four 0.26-0.34; "
        "other four mutually 0.73-0.90, supporting separate pathways.\n"
    )
```
