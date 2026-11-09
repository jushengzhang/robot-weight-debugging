Light Conditions showed no observed drop: 118/128 original versus 119/128 edited. I will now halve the strength of the same rank-two edit to test a dose-response, without changing the selected directions.
```python
def read_roll(k):
    return [json.loads(s) for s in (W/f"roll_{k}.jsonl").read_text().splitlines() if s.strip()]
def paired_summary(k, exclude_errors=False, category=None):
    rows=read_roll(k)
    by=defaultdict(dict)
    for r in rows:
        if category is not None and r["category"]!=category:
            continue
        by[(r["suite"],r["id"],r["seed"])][r["model"]]=r
    pairs=[p for p in by.values() if "original" in p and "edited" in p]
    if exclude_errors:
        pairs=[p for p in pairs if all(r.get("termination")!="error" for r in p.values())]
    d=np.array([int(p["edited"]["success"])-int(p["original"]["success"]) for p in pairs],float)
    return {"n":len(pairs), "original_success":sum(p["original"]["success"] for p in pairs),
            "edited_success":sum(p["edited"]["success"] for p in pairs),
            "diff_pp":float(100*d.mean()),"se_pp":float(100*d.std(ddof=1)/np.sqrt(len(d))),
            "gains":int((d==1).sum()),"losses":int((d==-1).sum())}

print("Full readout camera:",paired_summary(14))
print("Full readout light:",paired_summary(15))
print("Light excluding infrastructure errors:",paired_summary(15,True))
ledger=json.loads((W/"pass_ledger.json").read_text())
for k,cat in [(14,"Camera Viewpoints"),(15,"Light Conditions")]:
    ledger.append({"pass":k,"type":"rollout","edit":"readout_camera_rank2_full","category":cat,**paired_summary(k)})
(W/"pass_ledger.json").write_text(json.dumps(ledger,indent=2))
print("Half dose:",set_readout_projection(.5,"readout_camera_rank2_half"))
with notes.open("a") as f:
    f.write("\n## Passes 13-15: protected readout projection is promising\n"
            "- Full rank2 edit: weight-relative change 4.174%; protected-span and action-encoder mapping errors <6e-9.\n"
            "- Replay: camera-pair XYZ RMS .071218 -> .069914 (-1.83%); 13/19 pairs improve. Severe edit RMS .01168 vs light .00282; no general action shrinkage.\n"
            "- Discovery camera pass14: 64/128 original vs 72/128 edited; +6.25 pp, SE3.28, 13 gains/5 losses.\n"
            "- Light safety pass15: 118/128 vs119/128; +.78pp, SE2.07, 4 gains/3 losses; one original error, analyze sensitivity separately.\n"
            "- Next: fixed-subspace half-dose and matched negative control, then untouched camera validation plus all other categories.\n"
            "- Current active candidate half-dose; full candidate saved, not yet validated as best. Passes 15/40.\n")
```
