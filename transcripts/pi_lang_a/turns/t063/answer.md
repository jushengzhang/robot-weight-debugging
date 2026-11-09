Full substitution rescued all eight cheese-in-bowl seeds, versus four at half dose and zero without the new row. There were no diagnostic regressions. I will freeze the 18-row candidate, back it up, and prepare a matched-input probe plus the reserved confirmation tests.
```python
s28,pairs28=paired_summary(records(28))
print("Full18:",s28)
for key in sorted(newkeys):
    ss=[(a,b,d) for a,b,d in pairs28 if (a["suite"],a["id"])==key]
    print(key,"original",sum(a["success"] for a,b,d in ss),"full18",sum(b["success"] for a,b,d in ss),"/",len(ss))
# Record the accidentally omitted successful control; compare doses on common keys.
keys28={(a["suite"],a["id"],a["seed"]) for a,b,d in pairs28}
print("Omitted vs fixed panel:",[e for e in panel if (e["suite"],e["id"],e["seed"]) not in keys28])
for k in (26,27,28):
    _,pp=paired_summary(records(k))
    pp=[x for x in pp if (x[0]["suite"],x[0]["id"],x[0]["seed"]) in keys28]
    print("Common79 dose",k, "original",sum(a["success"] for a,b,d in pp),
          "edited",sum(b["success"] for a,b,d in pp))

(W/"candidate18_frozen.json").write_text(json.dumps(candidate18full,indent=2))
backup18=W/"candidate18_weights"; backup18.mkdir(exist_ok=True)
for p in (W/"edited").iterdir():
    if p.is_file(): shutil.copy2(p,backup18/p.name)

# Recover exact saved input paths without editing observations or prompts.
def first_input(r):
    d=Path(r["frames_dir"])
    p=d/"t0000_inputs.pt"
    if p.exists(): return str(p)
    pp=sorted(d.glob("*inputs.pt"))
    return str(pp[0]) if pp else None
diag_orig={(r["suite"],r["id"]):r for k in (1,2) for r in records(k)
           if r.get("instruction") and r.get("frames_dir")}
probe29_inputs=[first_input(diag_orig[key]) for key in
                (("libero_goal",1379),("libero_goal",1378),("libero_10",1466),("libero_10",1464))]
for suite in suites:
    rr=[r for r in records(19) if r["suite"]==suite and r["model"]=="original"
        and r.get("instruction") and r.get("frames_dir")]
    probe29_inputs.append(first_input(rr[0]))
assert all(probe29_inputs)
req29={"type":"probe","model":"both","inputs":probe29_inputs,
       "return":{"layers":[0,3,6,9,12,17],"vlm_layers":[0,3,6,9,12,17],"what":"resid"}}
(W/"request29.json").write_text(json.dumps(req29))
confirm_first=stage_b["libero_goal"]+stage_b["libero_10"]
confirm_second=stage_b["libero_object"]+stage_b["libero_spatial"]
# The remaining unused variants were never in either validation stage.
all_val_ids={(e["suite"],e["id"]) for e in validation}
outside=[v for v in unused if (v["suite"],v["id"]) not in all_val_ids]
confirm_third=[{"suite":v["suite"],"id":v["id"],"seed":2000+j} for j,v in enumerate(outside)]
confirmation={"first":confirm_first,"second":confirm_second,"third":confirm_third}
(W/"candidate18_confirmation.json").write_text(json.dumps(confirmation,indent=2))
print("Fresh confirmation sizes:",{k:len(v) for k,v in confirmation.items()})
notes=(W/"notes.md").read_text()
notes=re.sub(r"Forward passes used: \d+/50\.", "Forward passes used: 28/50.",notes)
notes=notes.replace("Current best: original weights (no edits).",
                    "Current best validated: candidate16. Active frozen candidate18 = candidate16 + spread->cheese,cup->mug full dose; awaiting fresh confirmation. Backups candidate16_weights and candidate18_weights. Only embedding table edited.")
notes+="\nPass28 full18: original48/79 edited64/79,+20.25pp(SE4.55),16rescues0regressions. One successful control(object1079seed89) accidentally omitted. On common79: base16=54,half18=60,full18=64; original48 each. Cheese1379 dose0/8->4/8->8/8. Mug1333 7/8->8/8, ambiguous microwave1466 stays0/8. Frozen18 now; no more lexical selection planned. Fresh confirmation manifest candidate18_confirmation.json contains113+128+113=354 episodes on unused variants, not diagnostic/stageA. StageA256 can be retested with final18 separately. Probe29 request saved. Six-category safety already checked candidate16; final18 needs canonical replay and safety confirmation. No validation instructions inspected for selection.\n"
(W/"notes.md").write_text(notes)
print(json.dumps(req29,separators=(",",":")))
```