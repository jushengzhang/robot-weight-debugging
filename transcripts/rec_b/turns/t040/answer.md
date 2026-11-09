The sham control preserved all 48 outcomes and the loop-failure count, so the paired execution paths are suitable for success comparisons. I will now ablate only layer 16 head 3’s output to test whether its border-focused lookup contributes causally to the failures.
```python
r16=read_roll(16)
pd16=pair_records(r16)
same_actions=0
for key,d in pd16.items():
    a=np.load(d["original"]["actions_npy"]); b=np.load(d["edited"]["actions_npy"])
    same_actions+=bool(a.shape==b.shape and np.array_equal(a,b))
print("sham exact action traces",same_actions,"/",len(pd16))
reset_edits()
edit_o_heads({16:{3:0.0}})
print("saved",list(edited()))
with (W/"notes.md").open("a") as f:
    f.write("\n## Sham control pass16\nIdentical weights:26/48->26/48; zero outcome discordance, "
            "loops21->21. One original failure trajectory varies in detailed flips across requests. "
            "Small weight edits still cause real but fragile outcome changes; require larger validation.\n")
```
