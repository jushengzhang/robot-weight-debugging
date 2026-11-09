The final edit gained 5.5 points on the repeated goal/long panel. I will audit the submitted episode IDs against the saved manifests before the next batch, so any transcription errors are recorded rather than silently treated as matched coverage.
```python
expected={17:batch17,18:batch18,26:panel,27:panel,28:panel,
          30:confirm_first,31:confirm_second,32:confirm_third,33:batch17}
audit={}
for k,ep in expected.items():
    want={(e["suite"],e["id"],e["seed"]) for e in ep}
    actual={(r["suite"],r["id"],r["seed"]) for r in records(k)}
    miss=sorted(want-actual); extra_ep=sorted(actual-want)
    audit[str(k)]={"missing":miss,"unexpected":extra_ep}
    if miss or extra_ep:
        print("Request audit",k,audit[str(k)])
(W/"request_audit33.json").write_text(json.dumps(audit,indent=2))
# Repair pass33's single ID transcription in the spare capacity of replication batch5.
repair33=[{"suite":s,"id":i,"seed":se} for s,i,se in audit["33"]["missing"]]
(W/"coverage_repair33.json").write_text(json.dumps(repair33,indent=2))
s33,_=paired_summary(records(33))
print("Pass33:",s33)
notes=(W/"notes.md").read_text()
notes=re.sub(r"Forward passes used: \d+/50\.", "Forward passes used: 33/50.",notes)
notes=notes.replace("Current best validated: candidate16.", "Current best validated: frozen candidate18.")
notes+="\nPass33 final18 stageA goal/long:101->108/128,+5.47pp(SE2.3),8rescues1regression. Request audit found a transcription error: long1314seed1003 was replaced by long1315seed1003 (both practice); do not claim exact610 unique coverage until repaired. coverage_repair33.json has missing episode; add it to replication batch5 (98+1 episodes) later. Other30-32 manifests matched. Pass28 missing control already recorded. Next34=request18 panel unchanged; then six safety batches. Frozen18 remains active.\n"
(W/"notes.md").write_text(notes)
print(json.dumps({"type":"rollout","model":"both","episodes":batch18},separators=(",",":")))
```