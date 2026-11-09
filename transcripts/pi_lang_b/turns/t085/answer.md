I’ll quantify how closely the direction-only control matches the full repair, including its movement toward the successful literal-instruction controls. Then I’ll temporarily remove just the three destination substitutions—“unit,” “furniture,” and “structure”—to test their contribution on the original development set.
```python
entries = json.loads((W/"probe_39"/"results.json").read_text())
probes[39] = {(r["idx"], r["arm"]): r for r in entries}
affected = [i for i in range(16) if not np.array_equal(action(23, i, "orig"), action(23, i, "edit"))]
d_final = np.concatenate([(action(23, i, "edit")-action(23, i, "orig")).ravel() for i in affected])
summary = {}
for label, k in [("norm_only", 38), ("direction_only", 39)]:
    d = np.concatenate([(action(k, i, "edit")-action(23, i, "orig")).ravel() for i in affected])
    summary[label] = {
        "displacement_norm_relative_to_final": float(np.linalg.norm(d)/np.linalg.norm(d_final)),
        "displacement_cosine_with_final": float(np.dot(d, d_final)/(np.linalg.norm(d)*np.linalg.norm(d_final))),
        "distance_from_final_relative_to_final_shift": float(np.linalg.norm(d-d_final)/np.linalg.norm(d_final)),
        "unchanged_controls": sum(np.array_equal(action(k, i, "edit"), action(k, i, "orig"))
                                  for i in [3, 5, 8, 10, 11])
    }
print("Direction/norm controls:", json.dumps(summary, indent=2))
print("Original probe reproducibility:",
      sum(np.array_equal(action(23, i, "orig"), action(39, i, "orig")) for i in range(16)), "/16")

pairs = [(0,3), (1,3), (2,3), (4,5), (6,8), (7,8), (9,10)]
print("\nAction distance to literal task control: input, original, norm, direction, full")
for i, j in pairs:
    control = action(23, j, "orig")
    vals = [np.linalg.norm(action(k, i, arm)-control)
            for k, arm in [(23,"orig"), (38,"edit"), (39,"edit"), (23,"edit")]]
    print(i, *[round(float(v), 4) for v in vals])

def feature(k, i, arm, key):
    with np.load(probes[k][i, arm]["activations"]) as z:
        return z[key].astype(np.float64)
print("\nCooktop residual distance to stove control, relative to original distance:")
for layer in [0, 3, 6, 9, 12, 15, 17]:
    vals = []
    for comp, region in [("vlm", "lang"), ("expert", "action")]:
        key = f"mean::{comp}.{layer}.resid::{region}"
        target = feature(23, 3, "orig", key)
        denom = np.linalg.norm(feature(23, 0, "orig", key)-target)
        vals.append((comp, [round(float(np.linalg.norm(feature(k, 0, "edit", key)-target)/denom), 3)
                           for k in [38, 39, 23]]))
    print(layer, vals)

(W/"causal_controls"/"direction_norm_results.json").write_text(json.dumps(summary, indent=2))
destination_rows = [4982, 16241, 5449]
base = load_original(ENAME)
keep_pos = [j for j, i in enumerate(row_ids) if i not in destination_rows]
base[torch.tensor([row_ids[j] for j in keep_pos])] = torch.from_numpy(final_rows[keep_pos])
save_edit(ENAME, base)
del base
gc.collect()
(W/"causal_controls"/"current_trial.json").write_text(json.dumps({
    "label": "destination_ablation", "restored_original_rows": destination_rows
}))
episodes40 = [{"suite": r["suite"], "id": r["id"], "seed": r["seed"]} for r in records[1]]
req40 = {"type": "rollout", "model": "both", "episodes": episodes40}
(W/"causal_controls"/"destination_ablation_request.json").write_text(json.dumps(req40, indent=2))
with (W/"notes.md").open("a") as f:
    f.write("\nPass39 direction/norm summary: " + json.dumps(summary) +
            "\nCURRENT EDIT TEMPORARY destination_ablation: final candidate minus rows4982,16241,5449. "
            "Pass40 planned original dev64, not used for candidate selection. Restore lex_dest_exact after40.\n")
print("\nPASS40 REQUEST:", json.dumps(req40, separators=(",", ":")))
```
