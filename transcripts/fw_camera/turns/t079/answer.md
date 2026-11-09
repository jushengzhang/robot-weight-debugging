The first untouched panel did not replicate the discovery gain: −1.6 points with a 2.7-point standard error. I will keep the edit unchanged for the already planned second panel, then judge the pooled confirmation result rather than tune against this result.
```python
print("Validation A:",paired_summary(20))
ledger=json.loads((W/"pass_ledger.json").read_text())
ledger.append({"pass":20,"type":"rollout","edit":"readout_camera_rank2_half",
               "purpose":"Untouched camera validation A",**paired_summary(20)})
(W/"pass_ledger.json").write_text(json.dumps(ledger,indent=2))
with notes.open("a") as f:
    f.write("\n## Pass20: first independent camera panel does not replicate\n"
            "- Validation A:59/128 original vs57/128 edited; -1.56pp SE2.71 (5 gains/7 losses).\n"
            "- Candidate unchanged for preplanned panel B. Do NOT promote based on discovery gain; original remains fallback.\n")
assert hashlib.sha256(load("mot.mixtures.action.head.weight").numpy().tobytes()).hexdigest()==frozen_hash
req21={"type":"rollout","model":"both","episodes":panels[1]}
(W/"request_21.json").write_text(json.dumps(req21))
print(json.dumps(req21,separators=(",",":")))
```
