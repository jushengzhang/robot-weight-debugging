Removing the destination substitutions reduces development success from 44/64 to 40/64; the original remains at 31/64. I’ll check which outcomes changed, restore the exact validated checkpoint, and inspect how the action expert reads the repaired language features. The ablation will not be submitted.
```python
def read_roll(k):
    return [json.loads(s) for s in (W/f"roll_{k}.jsonl").read_text().splitlines() if s.strip()]

def paired_rows(k):
    out = defaultdict(dict)
    for r in read_roll(k):
        out[(r["suite"], r["id"], r["seed"])][r["model"]] = r
    return out

p16, p40 = paired_rows(16), paired_rows(40)
discord = []
for key in sorted(set(p16) & set(p40)):
    a, b = p16[key]["edited"], p40[key]["edited"]
    if a.get("success") != b.get("success"):
        toks = sp.encode(a.get("instruction", ""), out_type=int)
        discord.append({
            "id": key[1], "full": a.get("success"), "without_dest": b.get("success"),
            "full_term": a.get("termination"), "ablation_term": b.get("termination"),
            "destination_tokens": [ascii_piece(i) for i in destination_rows if i in toks],
            "instruction": a.get("instruction")
        })
print("Destination ablation/full discordances:", json.dumps(discord, indent=2, ensure_ascii=True))
print("Roll16 terminations:", {arm: dict(__import__("collections").Counter(
    r["termination"] for r in read_roll(16) if r["model"] == arm)) for arm in ["original", "edited"]})

# Restore only the frozen, independently validated edit.
exact_array = np.load(final_path)
save_edit(ENAME, torch.from_numpy(exact_array))
del exact_array
gc.collect()
(W/"causal_controls"/"current_trial.json").write_text(json.dumps({
    "label": "FINAL_lex_dest_exact", "checkpoint": str(final_path), "rows": row_ids
}))

for name in ["validation_manifest.json", "validation_seed37_requests.json"]:
    obj = json.loads((W/name).read_text())
    print("\n", name, "type", type(obj).__name__)
    if isinstance(obj, dict):
        print("keys", list(obj))
        for key, val in obj.items():
            print(key, "length" if isinstance(val, (list, dict)) else "value",
                  len(val) if isinstance(val, (list, dict)) else val,
                  "preview", str(val)[:300])
    else:
        print("length", len(obj), "first", str(obj[:1])[:500])

old_attn = json.loads((W/"probe_3"/"results.json").read_text())
print("\nPrior attention metadata:", str(old_attn[0])[:2300])
sample_input = torch.load(probes[23][0, "orig"]["input"], map_location="cpu", weights_only=False)
print("Saved input keys:", list(sample_input))
for k, v in sample_input.items():
    if isinstance(v, dict):
        print(k, {kk: (str(type(vv).__name__), getattr(vv, "shape", None)) for kk, vv in v.items()})
    else:
        print(k, type(v).__name__, getattr(v, "shape", None),
              str(v)[:200] if isinstance(v, (str, int, float)) else "")
del sample_input

with (W/"notes.md").open("a") as f:
    f.write("\nPass40 destination ablation (17 rows):31->40/64,10 repairs/1 regression,+14.06pp SE4.9, "
            "versus full20 rows31->44/64. Full/ablation discordances: "+json.dumps(discord)+
            "\nRESTORED FINAL lex_dest_exact after40 using exact checkpoint. "
            "Cooktop residual mean distance to literal shrinks in VLM middle layers and expert; "
            "not all layers converge (VLM17 diverges), so do not claim global hidden-state matching. "
            "Expert residuals averaged over denoising steps are not a temporal localization claim.\n"
            "Remaining plan:41 final attention replay;42-47 each nonlanguage category difficulty4-5 n128 "
            "(sample.seed73 selects variants, rollout seed7);48-50 first384 frozen validation variants "
            "at explicit seed73. No further candidate selection.\n")
print("Final selected artifact restored:", edited())
```
