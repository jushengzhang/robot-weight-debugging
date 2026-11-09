Lighting showed two regressions, a -1.6-point difference with SE 1.1. I will check whether those paired runs started from identical observations, re-audit all 18 changed rows, and save canonical inputs for an exact replay check.
```python
source18={x["source_id"] for x in candidate18full}
assert not (source18 & protected)
Echeck=load(EMB)
rowchanges=[]; nvalues=0
for lo in range(0,E0.shape[0],8192):
    hi=min(lo+8192,E0.shape[0])
    neq=E0[lo:hi].to(torch.bfloat16)!=Echeck[lo:hi].to(torch.bfloat16)
    nvalues+=int(neq.sum())
    rowchanges+=(torch.where(neq.any(dim=1))[0]+lo).tolist()
assert set(rowchanges)==source18
print("Weight audit:",len(rowchanges),"rows;",nvalues,"differing bf16 values; protected overlap",source18 & protected)
del Echeck
print("Additional row geometry:")
for x in extra:
    a,b=E0[x["source_id"]],E0[x["target_id"]]
    print(x["source"],"->",x["target"],"target/source norm",round(float(b.norm()/a.norm()),5),
          "cosine",round(float(torch.dot(a,b)/(a.norm()*b.norm())),5))

final_rows=[r for k in (30,31,32,33,34) for r in records(k)]
sfirst,pfirst=paired_summary(final_rows)
print("Final18 broad language first pass:",sfirst,
      "distinct variants",len({(a["suite"],a["id"]) for a,b,d in pfirst}))
print("By difficulty:",{str(d):paired_summary([r for r in final_rows if r["difficulty"]==d])[0]
                        for d in sorted({r["difficulty"] for r in final_rows})})
(W/"final18_language_first_summary.json").write_text(json.dumps(sfirst,indent=2))
got34={(r["suite"],r["id"],r["seed"]) for r in records(34)}
assert got34=={(e["suite"],e["id"],e["seed"]) for e in batch18}

light_details=[]
for k in (35,36,37):
    s,pp=paired_summary(records(k))
    exp=sum(bool(source18 & set(sp.encode(a["instruction"]+"\n",out_type=int))) for a,b,d in pp)
    print("Final safety",k,s,"exposure",exp)
    if k==37:
        for a,b,d in pp:
            if d:
                ia=torch.load(first_input(a),map_location="cpu",weights_only=False)
                ib=torch.load(first_input(b),map_location="cpu",weights_only=False)
                dd={"suite":a["suite"],"id":a["id"],"difference":d,
                    "prompt_equal":ia["element"]["prompt"]==ib["element"]["prompt"],
                    "noise_equal":bool(np.array_equal(ia["noise"],ib["noise"]))}
                for cam in ("observation/image","observation/wrist_image"):
                    aa=np.asarray(ia["element"][cam]); bb=np.asarray(ib["element"][cam])
                    dd[cam]={"equal":bool(np.array_equal(aa,bb)),
                             "mean_abs_pixel_difference":float(np.abs(aa.astype(float)-bb.astype(float)).mean())}
                light_details.append(dd)
print("Lighting discordant-pair input audit:",json.dumps(light_details,indent=1))
(W/"light37_input_audit.json").write_text(json.dumps(light_details,indent=2))

canonical_inputs={}
for k in (37,19):
    for r in records(k):
        if r["model"]=="original" and r.get("instruction") and r.get("frames_dir"):
            key=(r["suite"],r["instruction"])
            if key not in canonical_inputs:
                assert not (source18 & set(sp.encode(r["instruction"]+"\n",out_type=int)))
                canonical_inputs[key]=first_input(r)
(W/"canonical_replay_inputs.json").write_text(json.dumps(list(canonical_inputs.values()),indent=2))
print("Canonical replay inputs:",len(canonical_inputs))
with (W/"notes.md").open("a") as f:
    f.write("\nPass34 final18 object/spatial124->124/128,0discordances. Broad first pass30-34:535->551/610 requested pairs (+2.623pp SE0.762),19rescues3regressions;609 distinct IDs pending repair. D5:63->80/109(+15.596pp). Final18 safety35 Camera101->101/128,36 InitialStates100->100/128;37 Light122->120/128 (-1.563pp SE1.10),zero edited-token exposure. First-input audit of lighting discordances saved light37_input_audit.json. Canonical replay set saved canonical_replay_inputs.json. Full inference-dtype audit:exactly18 changed rows, no protected-token intersection.\n")
```