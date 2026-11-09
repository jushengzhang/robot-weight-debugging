The layer-11 contrast is distributed across many channels and overlaps ordinary task variation, so removing its raw leading direction would be unsafe. I will assemble unmodified first-observation inputs from the existing discovery rollouts and canonical controls to identify contrast directions that are orthogonal to the successful task representations.
```python
r1=[json.loads(s) for s in (W/"roll_1.jsonl").read_text().splitlines()]
r2=[json.loads(s) for s in (W/"roll_2.jsonl").read_text().splitlines()]
tasks=json.loads((W/"task_index.json").read_text())
def task_of(rec):
    matches=[t["task"] for t in tasks if t["suite"]==rec["suite"]
             and rec["name"].startswith(t["task"])]
    if not matches:
        import re
        return re.split(r"_language_|_light_",rec["name"])[0]
    return max(matches,key=len)

bank=W/"discovery_input_bank"
bank.mkdir(exist_ok=True)
bank_records=[]
seen_inputs=set()
for rec in r1+r2+r3:
    path=(Path(rec["frames_dir"])/"t0000_inputs.pt").resolve()
    if str(path) in seen_inputs: continue
    seen_inputs.add(str(path))
    link=bank/f'{len(bank_records):03d}_{rec["suite"]}_v{rec["id"]}_inputs.pt'
    if not link.exists(): link.symlink_to(path)
    bank_records.append({**rec,"input":str(path),"link":str(link),"task":task_of(rec)})
(W/"discovery_input_manifest.json").write_text(json.dumps(bank_records,indent=2))
print("Unmodified inputs:",len(bank_records),"tasks",len(set((r["suite"],r["task"]) for r in bank_records)))
print("Successes:",sum(r["success"] for r in bank_records),
      "canonical controls:",sum(r["category"]!="Language Instructions" for r in bank_records))
print("Bank path:",bank)
print("Every symlink resolves to original rollout artifact:",
      all(Path(r["link"]).resolve()==Path(r["input"]) for r in bank_records))
```
