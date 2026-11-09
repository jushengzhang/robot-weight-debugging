Reducing padded-action input made performance worse, so that hypothesis is rejected. I will revisit the head interventions using their loop and motion outcomes, not attention concentration alone. In particular, a fixed-patch head may carry useful global state; suppressing it can be the wrong intervention even when its attention looks suspicious.
```python
for n in list(edited()):drop_edit(n)
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass31: unused-padding attenuation rejected\n"
            "Half padded action-input columns:66/128->61/128 (-3.9 +/-2.0pp),1win/6losses, "
            "position7->5,task13->11,object14->13; loops28->36. "
            "No evidence that unused noise drives the failure; original restored. No stronger dose justified.\n")
def paired_records(k):
    rs=[json.loads(x) for x in (W/f"roll_{k}.jsonl").read_text().splitlines()]
    dd=collections.defaultdict(dict)
    for r in rs:dd[(r["id"],r["seed"])][r["model"]]=r
    return [v for v in dd.values() if "original" in v and "edited" in v]
print("Earlier head interventions:")
for k in [12,16,22,23,26]:
    pairs_r=paired_records(k)
    o=[r["original"] for r in pairs_r];e=[r["edited"] for r in pairs_r]
    common=[r for r in pairs_r if not r["original"]["success"] and not r["edited"]["success"]]
    print("pass",k,"n",len(o),"success",sum(r["success"] for r in o),"->",sum(r["success"] for r in e),
          "loop failures",sum(not r["success"] and r["grip_flips"]>=6 for r in o),"->",
          sum(not r["success"] and r["grip_flips"]>=6 for r in e),
          "common-fail mean flips",
          round(np.mean([r["original"]["grip_flips"] for r in common]),1),"->",
          round(np.mean([r["edited"]["grip_flips"] for r in common]),1))
# Temporal versus pure-language sensitivity in the existing controlled maps.
for l in [13,14,16,17]:
    a0=np.load(W/"probe_21"/f"probe_orig_0_attn_L{l}.npy").mean(1)
    a1=np.load(W/"probe_21"/f"probe_orig_1_attn_L{l}.npy").mean(1)
    b0=np.load(W/"probe_21"/f"probe_edit_0_attn_L{l}.npy").mean(1)
    print("Layer",l,"temporal TV",np.round(np.abs(a1-a0).sum(1)/2,3).tolist(),
          "language TV",np.round(np.abs(b0-a0).sum(1)/2,3).tolist())
```
