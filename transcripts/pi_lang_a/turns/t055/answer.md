All six safety categories have now been tested. I will audit token exposure and action equality across them, inspect the background-error records, and check the tokenization of a few unresolved diagnostic phrases before deciding whether a small extension is justified.
```python
safety_audit={}
for k in range(19,25):
    rr=records(k); s,pairs=paired_summary(rr)
    exp=sum(bool(source16 & set(sp.encode(a["instruction"]+"\n",out_type=int))) for a,b,d in pairs)
    full_equal=first_equal=checked=0
    diffs=[]
    for a,b,d in pairs:
        pa,pb=a.get("actions_npy"),b.get("actions_npy")
        if pa and pb and Path(pa).exists() and Path(pb).exists():
            aa,bb=np.load(pa),np.load(pb); checked+=1
            first_equal+=int(np.array_equal(aa[:5],bb[:5]))
            full_equal+=int(np.array_equal(aa,bb))
        if d:
            diffs.append((a["suite"],a["id"],d,a["termination"],b["termination"]))
    safety_audit[str(k)]={"summary":s,"exposed":exp,"actions_checked":checked,
                         "first5_exact":first_equal,"entire_trajectory_exact":full_equal,
                         "discordances":diffs}
    print("SAFETY",k,s,"exposure",exp,"first5/full exact:",first_equal,full_equal,"/",checked,"diffs",diffs)
print("\nBackground failed-start records:")
r22=records(22)
for r in r22:
    if r.get("termination")=="error":
        print({k:v for k,v in r.items() if k not in ("actions_npy","frames_dir")})
(W/"safety_audit24.json").write_text(json.dumps(safety_audit,indent=2))

print("\nSpace-prefixed lexical fragments, diagnostic occurrences:")
terms=("concave","spreadable","spread","cooktop","cook","hob","activate","engage","merlot",
       "heating","appliance","microwave","cup","vessel","saucer","dish","storage","structure")
for text in terms:
    ids=sp.encode(" "+text,out_type=int)
    occ=[(r["suite"],r["id"],r["success"]) for r in diag if text in r["instruction"].split()]
    print(text,[(i,sp.id_to_piece(i).replace("\u2581","_"),i in protected) for i in ids],
          "diag_occ",occ[:15])
print("\nDetailed unresolved lexical phrases from diagnostics:")
for r in diag:
    if any(t in r["instruction"] for t in ("spreadable","merlot","heating food","electrical cooking","concave")):
        print(r["suite"],r["id"],"baseline_success",r["success"],r["instruction"])

print("\nProbe3 attention metadata:")
p3=json.loads((W/"probe_3"/"results.json").read_text())
print(json.dumps(p3[0] if isinstance(p3,list) else p3,indent=1)[:2600])
with (W/"notes.md").open("a") as f:
    f.write("\nPass21-24 safety: Light raw112/128 both; Background119->116/128 (3 extra edit-arm errors); Sensor113/128 both; Layout108->109/128. Detailed token/action audit in safety_audit24.json. Keep raw and complete-case metrics separate; infrastructure errors not evidence for/against policy. No safety prompts should read the edited rows.\n")
```