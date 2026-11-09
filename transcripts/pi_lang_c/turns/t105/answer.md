I will check the replay results element by element and look for genuine same-task controls among the saved language-validation inputs. This will determine what the final mechanism probe can support without modifying any inputs.
```python
meta49 = json.loads((W / "probe_49" / "results.json").read_text())
by49 = collections.defaultdict(dict)
for r in meta49:
    by49[r["idx"]][r["arm"]] = r
assert len(by49) == 60
audit49 = []
for idx, pair in sorted(by49.items()):
    assert set(pair) == {"orig", "edit"}
    ao, ae = np.load(pair["orig"]["action"]), np.load(pair["edit"]["action"])
    with np.load(pair["orig"]["activations"]) as xo, np.load(pair["edit"]["activations"]) as xe:
        assert set(xo.files) == set(xe.files)
        unequal = [key for key in xo.files if not np.array_equal(xo[key], xe[key])]
        n_arrays = len(xo.files)
    audit49.append({"idx": idx, "action_exact": bool(np.array_equal(ao, ae)),
                    "action_max_abs_diff": float(np.max(np.abs(ao - ae))),
                    "activation_arrays": n_arrays, "unequal_activation_keys": unequal,
                    "both_match_saved_chunk": all(pair[a]["matches_saved_chunk"] for a in ["orig", "edit"])})
summary49 = {
    "n_inputs": len(audit49),
    "exact_action_chunks": sum(x["action_exact"] for x in audit49),
    "exact_activation_sets": sum(not x["unequal_activation_keys"] for x in audit49),
    "max_action_difference": max(x["action_max_abs_diff"] for x in audit49),
    "activation_arrays_per_input": sorted(set(x["activation_arrays"] for x in audit49)),
    "both_arms_match_saved": sum(x["both_match_saved_chunk"] for x in audit49)}
print("FIXED-INPUT SAFETY:", json.dumps(summary49))
print("ANY MISMATCHES:", [x for x in audit49 if not x["action_exact"] or x["unequal_activation_keys"]])
(W / "F_final_safety_probe_audit.json").write_text(json.dumps({
    "summary": summary49, "inputs": audit49}, indent=2))
with (W / "notes.md").open("a") as f:
    f.write("\n## Pass49 exact-input safety\n" + json.dumps(summary49) +
            "\nIncludes both discordant episodes, both source trajectories at steps0,50,100; "
            "six categories and four suites covered. No input files modified.\n")

def p35_input(r):
    return (W / "roll_35_frames" /
            f"{r['suite']}_v{r['id']}_s{r['seed']}_original" / "t0000_inputs.pt")

def task_stem(r):
    return r["name"].split("_language_")[0]

r35_orig = [r for r in read_roll(35) if r["model"] == "original"]
p35_bykey = {(p["suite"], p["id"]): p for p in paired_roll(35)}
recovered35 = [r for r in r35_orig
               if not p35_bykey[(r["suite"], r["id"])]["original"]
               and p35_bykey[(r["suite"], r["id"])]["edited"]]
print("NATURAL SAME-TASK CONTROL SEARCH:")
for r in recovered35:
    x = torch.load(p35_input(r), map_location="cpu", weights_only=False)
    candidates = [q for q in r35_orig if q["suite"] == r["suite"]
                  and task_stem(q) == task_stem(r)
                  and not (set(tok.encode(q["instruction"] + "\n", add_bos=True)) & set(changed))]
    details = []
    for q in candidates:
        y = torch.load(p35_input(q), map_location="cpu", weights_only=False)
        imgs = ["observation/image", "observation/wrist_image"]
        fraction = [float(np.mean(x["element"][k] != y["element"][k])) for k in imgs]
        details.append({"id": q["id"], "instruction": q["instruction"],
                        "image_changed_fraction": fraction,
                        "noise_equal": bool(np.array_equal(x["noise"], y["noise"])),
                        "outcomes": [p35_bykey[(q["suite"], q["id"])][a] for a in ["original", "edited"]]})
    print(json.dumps({"suite": r["suite"], "id": r["id"], "instruction": r["instruction"],
                      "task": task_stem(r), "untouched_controls": details}, ensure_ascii=True))

meta31 = json.loads((W / "probe_31" / "results.json").read_text())
print("ATTENTION METADATA SCHEMA:", str(meta31[0])[:2200])
```
