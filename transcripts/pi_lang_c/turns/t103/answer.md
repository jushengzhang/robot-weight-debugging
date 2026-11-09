After deduplication, the safety checks cover 1,388 variant-seed pairs; the largest observed category decrease is 0.42 points. I will select inputs from all six categories and all four suites, plus both discordant episodes, to compare the two models on exactly the same images, prompts, and noise.
```python
safety_probe_cases = []
seen_paths = set()

def add_safety_probe(k, r, arm, step, reason):
    path = (W / f"roll_{k}_frames" /
            f"{r['suite']}_v{r['id']}_s{r['seed']}_{arm}" /
            f"t{step:04d}_inputs.pt")
    assert path.is_file(), str(path)
    if str(path) in seen_paths:
        return
    payload = torch.load(path, map_location="cpu", weights_only=False)
    prompt = payload["element"]["prompt"]
    token_ids = tok.encode(prompt.strip().lower().replace("_", " ") + "\n", add_bos=True)
    assert not (set(token_ids) & set(changed))
    seen_paths.add(str(path))
    safety_probe_cases.append({"path": str(path), "category": r["category"],
                               "suite": r["suite"], "id": r["id"],
                               "step": step, "source_arm": arm, "reason": reason})

for k in range(43, 49):
    rs = [r for r in read_roll(k) if r["model"] == "original"]
    for suite in ["libero_spatial", "libero_object", "libero_goal", "libero_10"]:
        candidates = [r for r in rs if r["suite"] == suite and r["steps"] >= 55]
        candidates.sort(key=lambda r: (-(r["difficulty"] or 0), r["id"]))
        r = candidates[0]
        for step in [0, 50]:
            add_safety_probe(k, r, "original", step, "suite-category coverage")

for p in new_discordant:
    r = next(r for r in read_roll(p["pass"]) if
             (r["suite"], r["id"], r["seed"], r["model"]) ==
             (p["suite"], p["id"], p["seed"], "original"))
    for arm in ["original", "edited"]:
        for step in [0, 50, 100]:
            add_safety_probe(p["pass"], r, arm, step, "discordant rollout audit")

request49 = {"type": "probe", "model": "both",
             "inputs": [c["path"] for c in safety_probe_cases],
             "return": {"layers": [0, 6, 12, 17],
                        "vlm_layers": [0, 6, 12, 17],
                        "vision_layers": [0, 13, 26], "what": "resid"}}
(W / "F_safety_probe_cases.json").write_text(json.dumps(safety_probe_cases, indent=2))
(W / "F_request_49.json").write_text(json.dumps(request49))
print("SAFETY PROBE INPUT COUNT:", len(safety_probe_cases))
print("INPUTS RELATIVE TO WORK:")
print(json.dumps([str(Path(c["path"]).relative_to(W)) for c in safety_probe_cases], indent=1))
print("RETURN:", request49["return"])
```
