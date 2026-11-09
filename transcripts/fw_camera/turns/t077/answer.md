The negative control gained only 1.6 points with a 2.5-point standard error. I will now restore and freeze the half-strength camera-subspace edit, then evaluate it on camera variants that have not been used for diagnosis, direction selection, or dose selection.
```python
r18=json.loads((W/"probe_18"/"results.json").read_text())
a18=np.stack([np.load(r["action"]) for r in r18]).reshape(19,3,32,7)
s18=action_probe_summary(orig_actions,a18)
print("Negative-control probe:",json.dumps(s18,separators=(",",":")))
print("Negative-control rollout:",paired_summary(19))
ledger=json.loads((W/"pass_ledger.json").read_text())
ledger += [{"pass":18,"type":"probe","edit":"readout_rank2_negative_control","inputs":57,**s18},
           {"pass":19,"type":"rollout","edit":"readout_rank2_negative_control",**paired_summary(19)}]
(W/"pass_ledger.json").write_text(json.dumps(ledger,indent=2))

set_readout_projection(.5)
frozen_hash=hashlib.sha256(load("mot.mixtures.action.head.weight").numpy().tobytes()).hexdigest()
(W/"candidate_lock.json").write_text(json.dumps({
    "candidate":"readout_camera_rank2_half","frozen_after_pass":19,
    "tensor_sha256":frozen_hash,"selection_rule":"Smallest tested targeted edit with comparable discovery gain",
    "alpha":.5,"rank":2,"validation_not_yet_seen":True},indent=2))
used_keys={(r["suite"],r["id"]) for p in W.glob("roll_*.jsonl")
           for r in [json.loads(s) for s in p.read_text().splitlines() if s.strip()]}
valcams=[v for v in pv if v["category"]=="Camera Viewpoints" and
         (v["suite"],v["id"]) in validation_keys and (v["suite"],v["id"]) not in used_keys]
valcams=sorted(valcams,key=lambda v:hashlib.sha256(f"camera_confirmation_2026:{v['suite']}:{v['id']}".encode()).hexdigest())
valeps=[{"suite":v["suite"],"id":v["id"],"seed":90000+i} for i,v in enumerate(valcams)]
panels=[valeps[i:i+128] for i in range(0,len(valeps),128)]
(W/"camera_validation_panels.json").write_text(json.dumps(panels,indent=2))
(W/"camera_validation_variants.json").write_text(json.dumps(valcams,indent=2))
print("Untouched camera panel sizes:",[len(p) for p in panels])
assert all((e["suite"],e["id"]) not in used_keys for p in panels for e in p)
assert len({(e["suite"],e["id"]) for p in panels for e in p})==len(valcams)
nt=notes.read_text().replace("- Forward passes used: 0 / 40.","- Forward passes used: 19 / 40; see chronological entries.",1)
nt=nt.replace("- Current best: original weights (no edits).",
              "- Frozen candidate: readout_camera_rank2_half. Original remains fallback pending independent validation.",1)
nt+="\n## Passes 18-19: negative control and candidate freeze\n"
nt+=f"- Norm-matched random rank2 control camera-pair RMS {s18['edited_camera_pair_rms']:.6f} vs original .071218.\n"
nt+="- Negative-control rollout +1.56pp SE2.48 (6 gains/4 losses), versus targeted half +7.03pp SE2.97.\n"
nt+="- Restored HALF targeted readout edit and froze tensor hash in candidate_lock.json before any validation results.\n"
nt+=f"- Untouched camera practice validation panels: {[len(p) for p in panels]}; no overlap with any used variant. All other category validation still required.\n"
nt+="- Active edit ONLY mot.mixtures.action.head.weight. All earlier edits reverted. Passes19/40.\n"
notes.write_text(nt)
req20={"type":"rollout","model":"both","episodes":panels[0]}
(W/"request_20.json").write_text(json.dumps(req20))
print(json.dumps(req20,separators=(",",":")))
```
