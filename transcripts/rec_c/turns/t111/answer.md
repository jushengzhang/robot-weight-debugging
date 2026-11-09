The first clearance dose changed the intended behavior locally but gave only 7 wins versus 6 losses in rollouts, which is not convincing. I will test a fourfold dose of the same frozen rank-one edit. This is a causal dose test—not a new feature search—and I will check the larger dose on fixed inputs before running it in the simulator.
```python
design=torch.load(W/"clearance_value_edit_design.pt",weights_only=False)
nominal_clearance=0.60
d=design["feature_direction"].float()
dv=design["value_direction_per_nominal"].float()*nominal_clearance
v_orig=load_original(vname).clone()
save_edit(vname,v_orig+torch.outer(dv,d))
cc=W/"candidate_empty_clearance_060";cc.mkdir(exist_ok=True)
for p in (W/"edited").iterdir():
    if p.is_file():shutil.copy2(p,cc/p.name)
r38=read_roll(38)
pairs38=collections.defaultdict(dict)
for r in r38:pairs38[r["id"],r["grip"],r["seed"]][r["model"]]=r
print("development discordances")
for key,rr in pairs38.items():
    if len(rr)!=2:continue
    o=rr["original"];e=rr["edited"]
    if o["success"]!=e["success"]:
        print(key[:2],"win" if e["success"] else "loss",o["level"],
              "flips",o["grip_flips"],e["grip_flips"])
for key,rr in pairs38.items():
    if key[0] in [979,562]:
        print("diagnostic",key[:2],[(m,r["success"],r["grip_flips"],round(r["eef_net_m"],3))
                                     for m,r in rr.items()])
with (W/"notes.md").open("a") as f:
    f.write("\nPass38 clearance0.15 development:71 vs70/126,7 wins/6 losses (+0.8+/-2.9pp), "
            "loops43 vs45. No established gain. Next fixed-feature dose0.60 (4x), with probe before rollout.\n")
```
