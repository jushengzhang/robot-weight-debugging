Softening the late head reduced loop failures from 21 to 18, but the success gain was only one episode and is not convincing. I will inspect whether the border fixation actually weakened, then use a stronger head-output ablation to distinguish a causal late-head contribution from a harmless attention sink.
```python
p14=json.loads((W/"probe_14/results.json").read_text())
for idx,d in paired_probe(p14).items():
    print("input",idx)
    for arm,p in d.items():
        m,gr,ent=attn_stats(np.load(p["attention"]["16"]))
        a=np.load(p["action"])
        print(arm,"L16h3 border509",round(float(m[3,509]),3),
              "B/W/L",np.round(gr[3,[0,1,3]],3).tolist(),"effective",round(float(np.exp(ent[3])),1),
              "xyz5",a[:5,:3].mean(0).round(3).tolist(),"grip5",a[:5,6].round(2).tolist())
r15=read_roll(15)
print("changed outcomes:",[(key,d["original"]["success"],d["edited"]["success"])
                         for key,d in pair_records(r15).items()
                         if d["original"]["success"]!=d["edited"]["success"]])
reset_edits()
edit_o_heads({16:{3:0.0}})
print("Current: L16h3 output-only ablation; Q restored.")
with (W/"notes.md").open("a") as f:
    f.write("\n## Candidate F results, passes14-15\nQ16h3 x.65:26/48->27/48, +2.1 +/-4.7 pp, "
            "3 rescues/2 regressions, loops21->18. Not statistically convincing. "
            "Test output-only ablation of same head to establish whether its contribution is causal.\n")
```
