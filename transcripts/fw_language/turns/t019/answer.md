I will prepare—but not submit—exact rank-one intervention templates for the candidate video heads. Each template changes only one output mode within one head and leaves all orthogonal modes intact, which will make the later dose-response interpretable. I will then give the backend more startup time.
```python
templates = {}
template_summary = []
for expert in ("video","action"):
    for r in candidates[:12]:
        layer,head=r["layer"],r["head"]
        s,us,vs=ov_spectrum(expert,layer,head,vectors=True)
        name=f"mot.mixtures.{expert}.blocks.{layer}.cross_attn.o.weight"
        O=load_original(name)
        hslice=O[:,128*head:128*(head+1)]
        factors=[]
        for j in (0,1):
            uj=us[:,j].contiguous()
            rj=(uj@hslice).contiguous()
            factors.append({"u":uj,"r":rj,"s":float(s[j])})
        key=f"{expert}_L{layer}_H{head}"
        templates[key]={"name":name,"layer":layer,"head":head,"factors":factors}
        template_summary.append({
            "key":key, "top_singular":float(s[0]),
            "full_removal_relative_tensor_norm":float(factors[0]["r"].norm()/O.norm()),
            "channel1938_overlap":float(us[1938,0].abs()) if expert=="video" else None
        })
torch.save(templates,W/"rank_one_intervention_templates.pt")
(W/"rank_one_template_summary.json").write_text(json.dumps(template_summary,indent=2))
print("Unsubmitted intervention templates:",len(templates))
for r in template_summary[:8]:
    print(r["key"],"singular",round(r["top_singular"],3),
          "max relative parameter change",round(r["full_removal_relative_tensor_norm"],4),
          "channel1938 overlap",round(r["channel1938_overlap"],4))
print("Submitted edit list remains:",edited())
with (W/"notes.md").open("a") as f:
    f.write("\n- Prepared unsubmitted rank-one head-output intervention factors in "
            "rank_one_intervention_templates.pt. O_head' = O_head - alpha*u*(u^T O_head), "
            "where u is an effective OV left singular mode. This is closed-form "
            "weight analysis, not fitting. Factors include second-mode controls. "
            "No intervention will be selected without behavior. Backend still offline.\n")
time.sleep(420)
print("Additional backend startup pause finished.")
```
