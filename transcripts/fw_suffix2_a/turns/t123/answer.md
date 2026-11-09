The active tensor exactly matches the frozen snapshot. I will register the remaining eleven evaluations and prepare the first fresh tagged batch. The retention rule stays fixed: a positive paired 95% confidence interval for tagged success, with clean and paraphrase point losses no greater than two percentage points.
```python
exec((W / "candidate_helpers.py").read_text())

plan = {}
for i in range(4):
    plan[40 + i] = {"split": f"tag_validation_{2+i}", "language": "raw", "condition": "tagged"}
for i in range(3):
    plan[44 + i] = {"split": f"tag_validation_{2+i}", "language": "strip", "condition": "clean"}
for i in range(4):
    plan[47 + i] = {"split": f"language_validation_{i}", "language": "raw", "condition": "paraphrase"}

def variant_keys(items):
    return {(x["suite"], int(x["id"])) for x in items}

used = set().union(*[
    variant_keys(splits[k]) for k in
    ["diagnostic", "language_diag", "development", "language_dev",
     "tag_validation_0", "tag_validation_1"]
])
fresh_keys = set()
for k in [f"tag_validation_{i}" for i in range(2, 6)] + [
    f"language_validation_{i}" for i in range(4)
]:
    keys = variant_keys(splits[k])
    assert len(keys) == 128 and not (keys & used) and not (keys & fresh_keys), k
    fresh_keys |= keys

manifest = json.loads((W / "frozen_video_orthogonal_rank6_manifest.json").read_text())
for filename, digest in manifest.items():
    assert hashlib.sha256((frozen_dir / filename).read_bytes()).hexdigest() == digest
assert list(edited()) == ["mot.mixtures.video.text_embedding.0.weight"]

(W / "final_rank6_validation_plan.json").write_text(json.dumps(plan, indent=2))

def request_for(passno):
    p = plan[passno]
    return {
        "type": "rollout", "model": "both", "language": p["language"],
        "episodes": [{k: x[k] for k in ("suite", "id", "seed")}
                     for x in splits[p["split"]]]
    }

def print_request(passno):
    req = request_for(passno)
    (W / f"request_{passno}.json").write_text(json.dumps(req, indent=2))
    print(json.dumps(req, separators=(",", ":")))

def log_validation(passno):
    result = paired_rollout(passno)
    condition = plan[passno]["condition"]
    completed = [p for p in plan if p <= passno and
                 plan[p]["condition"] == condition and (W / f"roll_{p}.jsonl").exists()]
    pooled = paired_rollout(completed)
    text = (
        f"\n### Pass{passno} frozen rank6 independent {condition}: {plan[passno]['split']}\n"
        + json.dumps(result) + "\nPooled " + condition + ": " + json.dumps(pooled)
        + "\nNo weight changes.\n"
    )
    with (W / "notes.md").open("a") as f:
        f.write(text)
    print("BATCH:", json.dumps(result))
    print("POOLED:", json.dumps(pooled))
    return result, pooled

print("Independent variants verified:", len(fresh_keys))
print("PASS40 REQUEST:")
print_request(40)
```
