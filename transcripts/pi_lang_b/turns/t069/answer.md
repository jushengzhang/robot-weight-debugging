I will check both action and internal-activation equality on the light replays. Then I will start the second-seed replication of all 512 reserved language variants, using the exact-source artifact and making no further edit selections.
```python
rr32,pi32=probe_index(32)
eq_actions=[]
eq_acts=[]
max_action_diff=0.0
for i in range(22):
    x,y=action_at(pi32,"orig",i),action_at(pi32,"edit",i)
    eq_actions.append(np.array_equal(x,y))
    max_action_diff=max(max_action_diff,float(np.max(np.abs(x-y))))
    with np.load(pi32[("orig",i)]["activations"]) as a, np.load(pi32[("edit",i)]["activations"]) as b:
        eq_acts.append(all(np.array_equal(a[k],b[k]) for k in a.files))
print("Light replay action equality:",Counter(eq_actions),"maxabs:",max_action_diff)
print("Light replay all requested activation equality:",Counter(eq_acts))
print("Original-arm saved chunk replay matches:",Counter(r.get("matches_saved_chunk") for r in rr32 if r["arm"]=="orig"))
with (W/"notes.md").open("a") as f:
    f.write(f"\nPass32 light cross-replay: {sum(eq_actions)}/22 original/edited action chunks bit-identical; "
            f"{sum(eq_acts)}/22 all requested VLM/expert residual arrays bit-identical. "
            "Initial light render images differed by1-16 channel entries (max1 uint8) for several same-seed paired runs, "
            "while robot state and flow noise were exact. Light discordances are not a same-input weight effect. "
            "All 253 disjoint light pairs:232->229 (-1.19pp SE1.05); severe new125:115->115.\n")
print("\nSEED37 VALIDATION REQUEST 1:")
print(json.dumps(val37[0],separators=(",",":")))
```
