For the bowl-and-plate paraphrase, the edit reduces action RMS distance to the successful literal reference from 0.202 to 0.025; the unexposed reference is unchanged. I will split the unused variants into an initial validation set and a reserved confirmation set, then test the frozen candidate on the first 128 paired episodes.
```python
# Stage the already-randomized validation manifest before seeing outcomes.
suites = sorted({e["suite"] for e in validation})
stage_a = {s:[e for e in validation if e["suite"]==s][:64] for s in suites}
stage_b = {s:[e for e in validation if e["suite"]==s][64:] for s in suites}
batch17 = stage_a["libero_goal"] + stage_a["libero_10"]
batch18 = stage_a["libero_object"] + stage_a["libero_spatial"]
plan = {"stage_a_17":batch17,"stage_a_18":batch18,
        "stage_b_reserved":[e for s in suites for e in stage_b[s]]}
(W/"validation16_stages.json").write_text(json.dumps(plan,indent=2))
# Candidate backup: save only the edited tensor file, preserving exact submitted bytes.
import shutil
backup=W/"candidate16_weights"
backup.mkdir(exist_ok=True)
for p in (W/"edited").iterdir():
    if p.is_file():
        shutil.copy2(p,backup/p.name)
with (W/"notes.md").open("a") as f:
    f.write("\nPre-validation staging: stageA64/suite (256 total), stageB remaining241 reserved. StageA batches17(goal+long),18(object+spatial). Candidate16 backed up unchanged. Probe16 bowl/plate action RMS to literal0.20231->0.02495; expert L17 RMS6.451->2.216. Cream-cheese description weaker0.27563->0.23912. Unexposed references bit-identical.\n")
request17={"type":"rollout","model":"both","episodes":batch17}
(W/"request17.json").write_text(json.dumps(request17))
print(json.dumps(request17,separators=(",",":")))
```