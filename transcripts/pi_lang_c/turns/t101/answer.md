The two safety sweeps total 1,536 paired episodes, with no overall change and small category-level differences. I will audit every discordance, check its actual instruction tokens, and inspect the saved inputs before selecting the fixed-input probes.
```python
safety_passes = list(range(24, 30)) + list(range(43, 49))
safety_pairs = sum([paired_roll(k) for k in safety_passes], [])
assert all(p["valid"] for p in safety_pairs)
safety_summary = {
    category: paired_summary([p for p in safety_pairs if p["category"] == category])
    for category in safety_categories}
safety_summary["overall"] = paired_summary(safety_pairs)
print("POOLED SAFETY:")
print(json.dumps(safety_summary, indent=2))

all_safety_texts = set()
for k in safety_passes:
    for r in read_roll(k):
        all_safety_texts.add(r["instruction"])
overlaps = {
    t: sorted(set(tok.encode(t.strip().lower().replace("_", " ") + "\n", add_bos=True)) & set(changed))
    for t in all_safety_texts}
assert not any(overlaps.values())
print("ALL SAFETY PROMPTS:", len(all_safety_texts), "EDITED TOKEN OVERLAPS: 0")

new_discordant = [p for p in safety_pairs if p["pass"] >= 43 and p["original"] != p["edited"]]
print("NEW SAFETY DISCORDANCES:")
for p in new_discordant:
    rs = {r["model"]: r for r in read_roll(p["pass"])
          if (r["suite"], r["id"], r["seed"]) == (p["suite"], p["id"], p["seed"])}
    print(json.dumps({**p, "instruction": rs["original"]["instruction"]}))
    a = np.load(rs["original"]["actions_npy"])
    b = np.load(rs["edited"]["actions_npy"])
    m = min(len(a), len(b))
    first_bad = np.flatnonzero(np.any(a[:m].reshape(m, -1) != b[:m].reshape(m, -1), axis=1))
    print("ACTION TRACES:", a.shape, b.shape,
          "first_unequal_action_index=", int(first_bad[0]) if len(first_bad) else None,
          "first_chunk_max_abs_diff=", float(np.max(np.abs(a[:5] - b[:5]))))
    folder = W / f"roll_{p['pass']}_frames"
    op = folder / f"{p['suite']}_v{p['id']}_s{p['seed']}_original" / "t0000_inputs.pt"
    ep = folder / f"{p['suite']}_v{p['id']}_s{p['seed']}_edited" / "t0000_inputs.pt"
    x = torch.load(op, map_location="cpu", weights_only=False)
    y = torch.load(ep, map_location="cpu", weights_only=False)
    def struct(z, depth=0):
        if torch.is_tensor(z) or isinstance(z, np.ndarray):
            return {"shape": list(z.shape), "dtype": str(z.dtype)}
        if isinstance(z, dict):
            return {str(k): struct(v, depth+1) for k, v in z.items()}
        if isinstance(z, (list, tuple)):
            return [struct(v, depth+1) for v in z[:3]]
        return str(z)[:180]
    print("INPUT SCHEMA:", json.dumps(struct(x), ensure_ascii=True)[:2100])
    def compare_leaves(a, b, path=""):
        diffs = []
        if isinstance(a, dict) and isinstance(b, dict):
            if set(a) != set(b):
                diffs.append((path, "different_keys"))
            for key in set(a) & set(b):
                diffs += compare_leaves(a[key], b[key], path + "/" + str(key))
        elif torch.is_tensor(a) or isinstance(a, np.ndarray):
            aa = a.detach().cpu().numpy() if torch.is_tensor(a) else a
            bb = b.detach().cpu().numpy() if torch.is_tensor(b) else np.asarray(b)
            if aa.shape != bb.shape:
                diffs.append((path, "shape_mismatch"))
            elif not np.array_equal(aa, bb):
                diffs.append((path, "changed_elements", int(np.count_nonzero(aa != bb))))
        elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            if len(a) != len(b):
                diffs.append((path, "length_mismatch"))
            for i, (aa, bb) in enumerate(zip(a, b)):
                diffs += compare_leaves(aa, bb, path + "/" + str(i))
        elif a != b:
            diffs.append((path, str(a)[:90], str(b)[:90]))
        return diffs
    print("INITIAL INPUT DIFFERENCES:", compare_leaves(x, y))

probe30_index = json.loads((W / "probe_30" / "index.json").read_text())
print("PROBE30 INDEX SCHEMA:", str(probe30_index)[:2000])
print("PASS35 RECOVERIES:")
r35_lookup = {(r["suite"], r["id"], r["model"]): r for r in read_roll(35)}
for p in paired_roll(35):
    if p["original"] != p["edited"]:
        rr = r35_lookup[(p["suite"], p["id"], "original")]
        print(json.dumps({**p, "instruction": rr["instruction"]}))

(W / "F_final_safety_validation.json").write_text(json.dumps({
    "summary": safety_summary, "episodes": safety_pairs,
    "unique_prompts": len(all_safety_texts), "edited_token_overlaps": 0}, indent=2))
with (W / "notes.md").open("a") as f:
    f.write("\n## Final rollout safety totals (passes24-29 and43-48)\n"
            + json.dumps(safety_summary) +
            "\nAll50 distinct safety prompts avoid every edited row. "
            "Two new discordances (robot-start and lighting) retained in totals; "
            "last probes will test exact-input invariance, including these episodes.\n")
```
