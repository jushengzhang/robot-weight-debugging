The 50% amplification also fails to improve backgrounds and introduces a light-control loss, so I’ll stop this readout family rather than increase its strength. The important distinction now is between texture features that are merely present and features associated with **reliable texture-induced failures**. I’ll identify those cases using original-model outcomes only, then inspect the action MLP pathway, which has not yet been tested.
```python
for n in list(edited()):
    drop_edit(n)
rows25 = paired_rows(25)
s25 = {c:paired_summary([r for r in rows25 if r["category"]==c])
       for c in sorted({r["category"] for r in rows25})}
(W/"roll25_summary.json").write_text(json.dumps(s25,indent=2))

# Original-only repeated outcomes; intervention outcomes do not select these cases.
original_outcomes = defaultdict(list)
for k in [2,13,16,21,22,23,24,25]:
    for line in (W/f"roll_{k}.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["model"] == "original":
            original_outcomes[(r["suite"],r["id"])].append(int(r["success"]))
rates = {k:float(np.mean(v)) for k,v in original_outcomes.items()}

failure_pairs = []
success_bg_indices = []
for i in range(54):
    if rates.get(source_keys[i],0) >= .75:
        success_bg_indices.append(i)
for hi,lo,li in triples:
    for bg,preferred in [(hi,lo),(lo,hi)]:
        if rates.get(source_keys[bg],1) <= .25:
            reference = preferred if rates.get(source_keys[preferred],0) >= .75 else li
            if rates.get(source_keys[reference],0) >= .75:
                failure_pairs.append([int(bg),int(reference)])

print("Amplification result:",s25)
print("Reliable failed/control initial-input pairs:")
for bg,ref in failure_pairs:
    print(source_keys[bg],rates[source_keys[bg]],"vs",
          source_keys[ref],rates[source_keys[ref]],
          "action RMS",round(rms(A0[bg]-A0[ref]),4))
print("Protected successful-background inputs:",len(success_bg_indices))

failure_manifest = {"failure_pairs":failure_pairs,
                    "successful_background_indices":success_bg_indices,
                    "original_outcome_passes":[2,13,16,21,22,23,24,25]}
(W/"failure_specific_manifest.json").write_text(json.dumps(failure_manifest,indent=2))
(W/"current_candidate.json").write_text(json.dumps({"family":"none"}))
with (W/"notes.md").open("a") as f:
    f.write("- Pass25 vertical amplification50%: "+json.dumps(s25)+". No benefit plus light regression; stop amplification for futility, without testing stronger amplification. All readout edits removed.\n")
    f.write("- Remaining15 passes reserved for one failure-specific MLP hypothesis and independent validation; no new hypothesis will be accepted without disjoint paired success evidence. Reliable failed/control cases selected solely from repeated ORIGINAL outcomes, saved in failure_specific_manifest.json.\n")
print("Active edits:",edited())
```
