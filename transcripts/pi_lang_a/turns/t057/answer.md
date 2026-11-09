I will summarize those attention maps and compare the earlier dairy-token edit with the safer frozen candidate. Two diagnostic failures suggest a narrow follow-up: map “spread” in cheese descriptions to the cheese embedding, and “cup” to mug, while leaving the ambiguous word “dairy” untouched.
```python
p25=json.loads((W/"probe_25"/"results.json").read_text())
P25={(r["arm"],r["idx"]):r for r in p25}
attn_summary={}
for idx in (0,2):
    pr=P25["orig",idx]
    saved=torch.load(pr["input"],map_location="cpu",weights_only=False)
    ids=[sp.bos_id()]+sp.encode(saved["element"]["prompt"]+"\n",out_type=int)
    layout=pr["prefix_layout"]; start=layout["lang"][0]; n=layout["n_lang_valid"]
    print("Attention input",idx,"token length",len(ids),n)
    srcpos=[start+i for i,t in enumerate(ids) if t in source16]
    lines=[]
    for layer in (3,6,9,15):
        a=np.load(P25["orig",idx]["attention"][str(layer)]).mean(axis=1)
        b=np.load(P25["edit",idx]["attention"][str(layer)]).mean(axis=1)
        heads=np.argsort(-a[:,start+1:start+n].sum(axis=1))[:2]
        for h in heads:
            entry={"layer":layer,"head":int(h),
                   "text_orig":round(float(a[h,start+1:start+n].sum()),4),
                   "text_edit":round(float(b[h,start+1:start+n].sum()),4),
                   "newline_orig":round(float(a[h,start+n-1]),4),
                   "newline_edit":round(float(b[h,start+n-1]),4),
                   "aliased_tokens_orig":round(float(a[h,srcpos].sum()),4),
                   "aliased_tokens_edit":round(float(b[h,srcpos].sum()),4)}
            lines.append(entry)
    attn_summary[str(idx)]=lines
    print(json.dumps(lines))
for a,b in ((0,1),(2,3)):
    ia=torch.load(P25["orig",a]["input"],map_location="cpu",weights_only=False)
    ib=torch.load(P25["orig",b]["input"],map_location="cpu",weights_only=False)
    eq={k:bool(np.array_equal(np.asarray(ia["element"][k]),np.asarray(ib["element"][k])))
        for k in ("observation/image","observation/wrist_image")}
    eq["noise"]=bool(np.array_equal(np.asarray(ia["noise"]),np.asarray(ib["noise"])))
    print("Matched reference",a,b,eq)
(W/"attention25_summary.json").write_text(json.dumps(attn_summary,indent=2))

print("\nEarlier diagnostic outcomes on cheese paraphrase and related controls:")
for k in (11,12,13,14,15):
    rr=records(k)
    selected=[(r["suite"],r["id"],r["model"],r["success"],r.get("termination"))
              for r in rr if r["id"] in (1379,1092,1079,1082,1466)]
    print(k,selected)

extra=[]
for source,target in (("spread","cheese"),("cup","mug")):
    si=sp.encode(" "+source,out_type=int); ti=sp.encode(" "+target,out_type=int)
    assert len(si)==len(ti)==1 and si[0] not in protected
    extra.append({"source":source,"target":target,"source_id":si[0],"target_id":ti[0],"alpha":1.0})
print("Proposed extension:",extra)
(W/"extension18_manifest.json").write_text(json.dumps(extra,indent=2))
extra_ids={x["source_id"] for x in extra}
new_exposed=[r for r in diag if extra_ids & set(sp.encode(r["instruction"]+"\n",out_type=int))]
print("New diagnostic exposure:",[(r["suite"],r["id"],r["success"]) for r in new_exposed])
seeds_extension=[7,13,23,37,53,71,89,107]
panel=[{"suite":r["suite"],"id":r["id"],"seed":s} for r in new_exposed for s in seeds_extension]
newkeys={(r["suite"],r["id"]) for r in new_exposed}
pilot=json.loads((W/"pilot_episodes.json").read_text())
controls=[e for e in pilot if (e["suite"],e["id"]) not in newkeys]
rng2=np.random.default_rng(18026)
controls=[controls[int(i)] for i in rng2.permutation(len(controls))[:24]]
panel+=controls
assert len(panel)<=128
(W/"extension_diagnostic_panel.json").write_text(json.dumps(panel,indent=2))
req26={"type":"rollout","model":"both","episodes":panel}
(W/"request26.json").write_text(json.dumps(req26))
with (W/"notes.md").open("a") as f:
    f.write("\nPass25 attention maps for two matched paraphrase/literal pairs: summary in attention25_summary.json. Considering only two additional semantically motivated rows (space-prefixed spread->cheese, cup->mug), selected from PRE-validation diagnostics; dairy stays unmodified. New dose panel saved extension_diagnostic_panel.json, candidate16 still active for baseline. StageB remains unused.\n")
print("BASELINE REQUEST:",json.dumps(req26,separators=(",",":")))
```