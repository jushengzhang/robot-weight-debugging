The first visual-subspace dose gained only 1/96, well within its paired uncertainty. Before trying a stronger dose, I will audit a possible numerical confound: many small edits have changed the same few outcomes, while the earlier sham used an fp32 tensor. I will check the stored precision and prepare an exactly unchanged bf16-tensor control.
```python
def paired_table(k):
    rr=read_jsonl(W/f"roll_{k}.jsonl")
    oo={(r["id"],r["grip"],r["seed"]):r for r in rr if r["model"]=="original"}
    ee={(r["id"],r["grip"],r["seed"]):r for r in rr if r["model"]=="edited"}
    return [(key,oo[key],ee[key]) for key in oo.keys()&ee.keys()]

for k in [25,29,31,33,35,36,39]:
    pairs=paired_table(k)
    wins=[key[:2] for key,o,e in pairs if not o["success"] and e["success"]]
    losses=[key[:2] for key,o,e in pairs if o["success"] and not e["success"]]
    print(k,"rescues",sorted(wins),"regressions",sorted(losses))
print("V17 fraction not exactly representable in bf16:",
      float((v_orig!=v_orig.to(torch.bfloat16).float()).float().mean()))

torch.save({"name":content_name,"basis":content_basis,"alpha":.5,
            "tensor":load(content_name)},W/"candidate_visual_M05.pt")
for n in list(edited()): drop_edit(n)
save_edit(content_name,v_orig.clone())
print("Exact bf16-tensor sham saved; max float32 difference:",
      float((load(content_name)-v_orig).abs().max()))
with (W/"notes.md").open("a") as f:
    f.write("\n## M alpha.5, passes38-39\n"
            "Probe upstream VLM residuals exactly unchanged; translational changes~.0004-.0016, "
            "no broad gripper bias. Diagnostic57/96->58/96,+1.0+/-2.8pp, "
            "4 rescues/3 regressions;loops35->31. Not sufficient. "
            "Before stronger dose, check identical bf16 V17 overlay; previous sham was fp32 bias.\n")
```
