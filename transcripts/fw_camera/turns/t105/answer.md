The untouched reserve panel is positive (+4.7 points), but the combined independent estimate is still uncertain. I’ll log it and continue the predeclared seed panels, using variant-clustered uncertainty so repeated seeds are not treated as entirely independent evidence.
```python
def clustered_summary(pairs):
    out=paired_summary(pairs)
    dbar=np.mean([p["d"] for p in pairs])
    grouped=defaultdict(list)
    for p in pairs:
        grouped[p["key"][:2]].append(p["d"])
    G=len(grouped); N=len(pairs)
    score=np.array([np.sum(np.asarray(v)-dbar) for v in grouped.values()])
    cluster_se=100*np.sqrt(G/(G-1)*np.sum(score**2))/N if G>1 else float("nan")
    variant_means=np.array([np.mean(v) for v in grouped.values()])
    out.update({"variants":G,"cluster_se_pp":round(float(cluster_se),3),
                "equal_variant_diff_pp":round(float(100*variant_means.mean()),3),
                "equal_variant_se_pp":round(float(100*variant_means.std(ddof=1)/np.sqrt(G)),3)})
    return out

current_camera_passes=[20,21,22,23,28]
agg28=clustered_summary(paired_rows(current_camera_passes))
print("Independent camera, frozen candidate:",agg28)
selection_keys=set()
for k in (1,4,6,9,11,14,17,19,25,27):
    for line in (root/f"roll_{k}.jsonl").read_text().splitlines():
        r=json.loads(line)
        if r["category"]=="Camera Viewpoints":
            selection_keys.add((r["suite"],r["id"]))
print("Overlap between final evaluation variants and discovery/diagnostic variants:",
      len(selection_keys.intersection(all_eval_keys)))
assert not selection_keys.intersection(all_eval_keys)
ledger=json.loads((root/"pass_ledger.json").read_text())
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
