All 60 safety inputs produced bit-identical action chunks and recorded activations in both models, including the two discordant rollout cases. I also found natural language/control pairs with identical images and noise. I will enumerate these pairs for the final probe, which can test task-feature changes without visual or noise confounds.
```python
payload35 = { (r["suite"], r["id"]): torch.load(p35_input(r), map_location="cpu", weights_only=False)
             for r in r35_orig }
touched35 = {}
for r in r35_orig:
    touched35[(r["suite"], r["id"])] = bool(
        set(tok.encode(r["instruction"].strip().lower() + "\n", add_bos=True)) & set(changed))

def same_policy_observation_and_noise(x, y):
    return (np.array_equal(x["noise"], y["noise"]) and
            all(np.array_equal(x["element"][k], y["element"][k])
                for k in ["observation/image", "observation/wrist_image"]))

natural_pairs = []
for r in sorted(r35_orig, key=lambda r: (r["suite"], r["id"])):
    key = (r["suite"], r["id"])
    if not touched35[key]:
        continue
    controls = [q for q in r35_orig
                if q["suite"] == r["suite"] and task_stem(q) == task_stem(r)
                and not touched35[(q["suite"], q["id"])]
                and p35_bykey[(q["suite"], q["id"])]["original"]
                and p35_bykey[(q["suite"], q["id"])]["edited"]
                and same_policy_observation_and_noise(
                    payload35[key], payload35[(q["suite"], q["id"])])]
    if not controls:
        continue
    q = min(controls, key=lambda q: q["id"])
    outcomes = p35_bykey[key]
    natural_pairs.append({
        "suite": r["suite"], "id": r["id"], "control_id": q["id"],
        "source_path": str(p35_input(r)), "control_path": str(p35_input(q)),
        "source_instruction": r["instruction"], "control_instruction": q["instruction"],
        "original_success": outcomes["original"], "edited_success": outcomes["edited"],
        "images_equal": True, "noise_equal": True
    })

unique_paths50 = []
for pair in natural_pairs:
    for key in ["source_path", "control_path"]:
        if pair[key] not in unique_paths50:
            unique_paths50.append(pair[key])
for pair in natural_pairs:
    pair["source_idx"] = unique_paths50.index(pair["source_path"])
    pair["control_idx"] = unique_paths50.index(pair["control_path"])

print("NATURAL MATCHED PAIRS:", len(natural_pairs),
      "UNIQUE INPUTS:", len(unique_paths50))
print("BY SUITE:", dict(collections.Counter(p["suite"] for p in natural_pairs)))
print("BY OUTCOME:", dict(collections.Counter(
    ("recovered" if not p["original_success"] and p["edited_success"]
     else "regressed" if p["original_success"] and not p["edited_success"]
     else "both_success" if p["original_success"] else "both_fail")
    for p in natural_pairs)))
with np.load(by49[0]["orig"]["activations"]) as z:
    print("ACTIVATION KEY/SHAPE EXAMPLES:", [(k, list(z[k].shape)) for k in z.files[:14]])
    print("EXPERT KEY EXAMPLES:", [k for k in z.files if "expert" in k][:12])

request50 = {"type": "probe", "model": "both", "inputs": unique_paths50,
             "return": {"layers": list(range(18)), "vlm_layers": list(range(18)),
                        "vision_layers": [0, 13, 26], "what": "resid"}}
(W / "F_natural_matched_probe_pairs.json").write_text(json.dumps(natural_pairs, indent=2))
(W / "F_request_50.json").write_text(json.dumps(request50))
print("FINAL PROBE INPUTS RELATIVE TO WORK:")
print(json.dumps([str(Path(p).relative_to(W)) for p in unique_paths50], indent=1))
with (W / "notes.md").open("a") as f:
    f.write("\n## Final planned mechanism probe\n"
            "Natural matched pairs in frozen validation batch35: " + str(len(natural_pairs)) +
            " pairs, " + str(len(unique_paths50)) +
            " unique genuine inputs. Images and flow noise are elementwise equal within each "
            "source/control pair; only natural instruction wording differs. Include every "
            "available touched source with a successful untouched same-task control, not "
            "only recovered cases. Record all18 VLM/expert residual layers. F unchanged.\n")
```
