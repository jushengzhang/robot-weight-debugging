Some early rollout aliases are absent, so I’ll check the split against the saved diagnostic groups and discovery episode list instead. This is only a bookkeeping fix; the model and evaluation plan remain unchanged.
```python
selection_keys={(e["suite"],e["id"]) for e in screen_eps}
for fn in ("diagnostic_groups.json","broad_diagnostic_groups.json"):
    for g in json.loads((root/fn).read_text()):
        selection_keys.update((g["suite"],vid) for vid in g["ids"][:2])
overlap=selection_keys.intersection(all_eval_keys)
assert not overlap
print("Selection/evaluation overlap:",len(overlap))
print("Independent camera:",agg28)
ledger=json.loads((root/"pass_ledger.json").read_text())
if not any(e.get("pass")==28 for e in ledger):
    ledger.append({"pass":28,"type":"rollout","edit":"readout_camera_rank2_half",
                   "purpose":"Never-run camera reserve",**paired_summary(paired_rows([28]))})
(root/"pass_ledger.json").write_text(json.dumps(ledger,indent=2))
with (root/"notes.md").open("a") as f:
    f.write("\n## Pass28: untouched camera reserve\n"
            "- 41/128 ->47/128 (+4.688pp SE3.110;11 gains/5 losses).\n"
            f"- All602 selection-disjoint variants, one seed each: {agg28}.\n"
            "- Still not conclusive; continue fixed seed panels, no tuning. Only frozen head.weight active.\n")
print("NEXT REQUEST:",json.dumps(json.loads((root/"request_29.json").read_text()),separators=(",",":")))
```
