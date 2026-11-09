The initial-state result differs by one edited-arm simulator error. I will audit the changed rows and token exposure to distinguish policy effects from execution noise, and inspect only the original diagnostic failures for any remaining lexical mechanism worth testing.
```python
torch.set_num_threads(16)
candidate16=json.loads((W/"candidate16_frozen.json").read_text())
source16={x["source_id"] for x in candidate16}
canonical_tasks=sorted({v["name"].split("_language_")[0].replace("_"," ") for v in languages})
protected={t for text in canonical_tasks for t in sp.encode(text+"\n",out_type=int)}|{sp.bos_id()}
print("Canonical tasks:",len(canonical_tasks),"protected token IDs:",len(protected))
print("Candidate overlap:",source16 & protected)
print("Candidate token pieces:",[(x["source"],sp.id_to_piece(x["source_id"]),x["target"]) for x in candidate16])

E0=load_original(EMB)
Ecur=load(EMB)
changed=[]
for lo in range(0,E0.shape[0],8192):
    hi=min(lo+8192,E0.shape[0])
    mask=(E0[lo:hi].to(torch.bfloat16)!=Ecur[lo:hi].to(torch.bfloat16)).any(dim=1)
    changed += (torch.where(mask)[0]+lo).tolist()
print("Actual changed rows at inference dtype:",len(changed),changed)
assert set(changed)==source16
assert all(torch.equal(Ecur[x["source_id"]].to(torch.bfloat16),
                       E0[x["target_id"]].to(torch.bfloat16)) for x in candidate16)
del Ecur

summaries={}
for k in (17,18,19,20):
    rr=records(k)
    s,pairs=paired_summary(rr)
    exposed=sum(bool(source16 & set(sp.encode(a["instruction"]+"\n",out_type=int))) for a,b,d in pairs)
    dis=[{"suite":a["suite"],"id":a["id"],"difference":d,
          "exposed":bool(source16 & set(sp.encode(a["instruction"]+"\n",out_type=int))),
          "orig_term":a.get("termination"),"edit_term":b.get("termination")}
         for a,b,d in pairs if d]
    summaries[str(k)]={"summary":s,"exposed_pairs":exposed,"discordances":dis}
    print("Pass",k,s,"exposed:",exposed,"discordances:",dis)
(W/"validation_safety_summary20.json").write_text(json.dumps(summaries,indent=2))

print("\nRemaining DIAGNOSTIC failures only (not validation text):")
diag=[r for k in (1,2) for r in records(k) if r.get("instruction")]
p15={(a["suite"],a["id"]):b["success"] for a,b,d in pairs15}
for r in diag:
    key=(r["suite"],r["id"])
    if not r["success"] and not p15.get(key,False):
        print(key, "candidate15:",p15.get(key,"untested"),r["instruction"])
print("\nPotential lexical fragments, tokenizer only:")
for text in ("concave","spreadable","spread","cooktop","cook","hob","activate","engage","power",
             "saucer","porcelain","ceramic","container","vessel","casserole","ramekin",
             "soya","soy","sauce","dairy","beverage","tin","canned","preserved","pudding",
             "dessert","microwavable","appliance","heating","vent","receptacle","storage"):
    ids=sp.encode(text,out_type=int)
    print(text, [(t,sp.id_to_piece(t),"PROTECTED" if t in protected else "") for t in ids])
with (W/"notes.md").open("a") as f:
    f.write("\nPass18 stageA object/spatial orig121/128 edit120/128 (-0.78pp SE1.75); discordances audited in validation_safety_summary20.json. Pass19 Camera128 pairs orig84 edit88 (+3.125pp SE1.54), unequal simulator errors. Pass20 InitialStates128 orig94 edit93 (-0.78pp SE0.78), sole regression an edited-arm error. Actual inference-dtype weight audit: exactly16 full embedding-row replacements, zero protected canonical token overlap.\n")
```