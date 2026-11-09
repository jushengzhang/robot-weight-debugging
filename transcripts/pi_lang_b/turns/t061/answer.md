Light Conditions showed three regressions in 128 pairs, a −2.3-point difference that warrants investigation. I will check whether those prompts use any edited row and whether the action traces differ from the first query. This should help distinguish an off-target model effect from differing rollout observations or execution variability.
```python
for k in (25,26,27):
    ps=pair_roll(k)
    print("\nSafety pass",k,paired_stats(ps))
    hits=[]
    first_diffs=[]
    discord=[]
    for p in ps:
        a,b=p["original"],p["edited"]
        instr=a["instruction"] or b["instruction"]
        touched=set(sp.encode(instr+"\n",add_bos=True)) & set(spec) if instr else set()
        if touched: hits.append((a["suite"],a["id"],sorted(touched),instr))
        if a["actions_npy"] and b["actions_npy"]:
            x,y=np.load(a["actions_npy"]),np.load(b["actions_npy"])
            if len(x) and len(y):
                d=float(np.max(np.abs(x[:min(5,len(x),len(y))]-y[:min(5,len(x),len(y))])))
                first_diffs.append(d)
                if bool(a["success"]) != bool(b["success"]):
                    discord.append((a["suite"],a["id"],a["termination"],b["termination"],a["steps"],b["steps"],d,instr))
    print("Prompts touching edit:",hits)
    print("First-5 exact matches:",sum(d==0 for d in first_diffs),"/",len(first_diffs),
          "max first5 diff:",max(first_diffs,default=0))
    print("Outcome discordances:",discord)

print("\nCamera record fields:",list(read_roll(25)[0]))
p25=next(p for p in pair_roll(25) if bool(p["original"]["success"])!=bool(p["edited"]["success"]))
print("Camera discordant metadata:",
      [{k:v for k,v in p25[m].items() if k not in ("instruction","name")} for m in ("original","edited")])
print("Camera frame index files:",[p.name for p in (W/"roll_25_frames").iterdir()][:5])
camera_initial=list((W/"fwd"/"roll_25"/"artifacts").glob("*frames/t0000_inputs.pt"))
print("Saved camera initial inputs:",len(camera_initial))
# Preserve the validation retry rule: replace only incomplete pairs, regardless of retry outcome.
retry_map={(p["original"]["suite"],p["original"]["id"],p["original"]["seed"]):p for p in pair_roll(24)}
vfinal=[]
for p in vp:
    a=p["original"]; key=(a["suite"],a["id"],a["seed"])
    vfinal.append(p if is_complete(p) else retry_map[key])
print("\nReconciled seed23 reserved validation:",paired_stats(vfinal))
with (W/"notes.md").open("a") as f:
    f.write("\nPass24 retried all 25 validation infrastructure-error pairs, same seeds: 25/25 success each. "
            "Reconciled reserved validation: original451/512, edited465/512, +2.734pp, 14 fixes/0 regressions, SE~.72pp. "
            "Primary raw result was +2.34pp SE1.14, reported separately.\n"
            "Safety exact artifact: pass25 Camera 88->87/128; pass26 RobotInitial94->94/128; "
            "pass27 Light117->114/128. Investigating light discordances before accepting safety result. "
            "No edited rows occur in canonical tasks; identical-input probes must be distinguished from simulator variability.\n")
```
