The second third-seed batch also has two regressions and no repairs. The aggregate validation gain remains positive, but the weaker seed-73 result needs to be reported. I’ll inspect those two regressions, update the cumulative statistics, and prepare the last pass: 125 language trials plus the three camera error retries.
```python
p49 = paired_rows(49)
assert set(p49) == {(e["suite"], e["id"], e["seed"]) for e in requests_final[1]["episodes"]}
disc49 = []
for key, pair in p49.items():
    if bool(pair["original"]["success"]) != bool(pair["edited"]["success"]):
        prompt = pair["original"]["instruction"]
        hits = sorted(set(sp.encode(prompt+"\n", add_bos=True)) & row_set)
        first_actions = {}
        if all(r.get("actions_npy") for r in pair.values()):
            ao = np.load(pair["original"]["actions_npy"])
            ae = np.load(pair["edited"]["actions_npy"])
            first_actions = {
                "original_shape": list(ao.shape), "edited_shape": list(ae.shape),
                "first_action_equal": bool(np.array_equal(ao[0], ae[0])),
                "first_action_l2": float(np.linalg.norm(ao[0].astype(float)-ae[0].astype(float)))
            }
        disc49.append({
            "suite": key[0], "id": key[1],
            "original": pair["original"]["success"], "edited": pair["edited"]["success"],
            "instruction": prompt, "edited_tokens": [ascii_piece(i) for i in hits],
            **first_actions
        })
for key, pair in p49.items():
    assert key not in language
    language[key] = pair
    raw_language[key] = pair
seed73_pairs = {key: pair for key, pair in language.items() if key[2] == 73}
print("Pass49 discordances:", json.dumps(disc49, indent=2, ensure_ascii=True))
print("Cumulative reconciled validation:", json.dumps(pair_stats(language), separators=(",", ":")))
print("Seed73 so far:", json.dumps(pair_stats(seed73_pairs), separators=(",", ":")))

report["validation_after49"] = pair_stats(language)
report["validation_raw_after49"] = pair_stats(raw_language)
report["seed73_after49"] = pair_stats(seed73_pairs)
report["pending_passes"] = [50]
report["final_candidate"] = "lex_dest_exact"
report["limitations"] = [
    "Lexical row edits are context-insensitive; ambiguous descriptions remain a risk.",
    "Seed73 first256 language trials have 0 repairs and4 regressions; retained in all aggregate results.",
    "Same-input canonical policy outputs are invariant, but simulator/render variability can cause rollout discordances.",
    "Practice validation is not the benchmark held-out evaluation."
]
(W/"final_report.json").write_text(json.dumps(report, indent=2))
with (W/"notes.md").open("a") as f:
    f.write("\nPass49 seed73 reserved batch2:111->109/128,0 repairs/2 regressions, no errors. "
            "Discordances="+json.dumps(disc49)+
            "\nCumulative validation after49="+json.dumps(pair_stats(language))+
            "\nSeed73 first256="+json.dumps(pair_stats(seed73_pairs))+
            ". Keep all losses; no retuning. Pass50 fixed:125 language seed73 +3 camera error retries seed7. "
            "Final selected artifact remains lex_dest_exact, unchanged since40 restoration.\n")
print("PASS50 REQUEST:", json.dumps(requests_final[2], separators=(",", ":")))
```
