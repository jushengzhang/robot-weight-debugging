The video text-output edit gave no net gain on the 96 tagged development episodes and lost one of 32 paraphrase episodes. I won’t retain it on that evidence. I’ll test the value-only version next, since it produced similar replay correction without changing attention selection.
```python
rows37=[json.loads(x) for x in (W/"roll_37.jsonl").read_text().splitlines() if x.strip()]
print("record schema",str(rows37[0])[:2400])
clear_edits()
for layer in range(30):
    n=f"mot.mixtures.video.blocks.{layer}.cross_attn.v.weight"
    o=load_original(n)
    save_edit(n,o-(o@By)@Dy.T)
print("active edits",len(edited()),"video values only, dose 1")
with (W/"notes.md").open("a") as f:
    f.write("\nPass37 development video text-output only: tagged49->49/96 (0pp), language26->25/32 (-3.125pp). Not retained on development evidence. Restored pass35 value-only projector (30 tensors) for same development rollout38.\n")
```
